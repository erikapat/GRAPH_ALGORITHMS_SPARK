import networkx as nx
import pandas as pd
import numpy as np


class Plot:
    """
    Plot is an `abstract class`. You can see Plot as a convenient namespace where plotting utilities can be found. There
    are no object inherited from Plot (by definition of abstract class) and it has no properties. Its methods relate
    because, typically, the output of one of them is what you will require as input for another one.
    """

    def nodes_max_weight(G, weight_name="weight", bidirectional=True):
        """
        This returns a dictionary with the maximum weight going out (or both in and out) of each node in a networkx graph.
        These weights are typically used for node and edge selection by other methods of the Plot class.

        :param G: The input graph
        :type G: networkx graph

        :param weight_name: The name of the edge attribute that used as 'weight'. All edges in the graph must have that attribute!
        :type weight_name: str

        :param bidirectional: If true (default), the value is computed for all nodes going either in or out. If false, it is only
            computed for the edges leaving the node.
        :type bidirectional: bool

        :returns: A dictionary where the keys are the node keys and the values are the maximum weight computed by this.
        :rtype: dict
        """
        nmw = {}
        for u in G:
            nmw[u] = 0

        for u in G:
            for v in G[u]:
                w = G[u][v][weight_name]

                nmw[u] = max(nmw[u], w)

                if bidirectional:
                    nmw[v] = max(nmw[v], w)

        return nmw

    def layered_plot(
        G, pos, layout, head=None, scale=1, tail=None, weight_name="weight"
    ):
        """
        This plots the networkx graph using matplotlib through networkx.

        :param G: A graph
        :type G: networkx graph

        :param pos: A networkx positioning of the nodes. Unlike in networkx, this parameter has no default value. You may use
            Graph.layered_positions() or any networkx function.
        :type pos: dict

        :param layout: See layout_reference_
        :type layout: list of (edgecol, nodecol, fontcol, wid, diam, height, above, radius) tuples

        :param head: A function (typically with matplotlib settings) run before starting the plot
        :type head: function

        :param scale: If other than 1, a way to globally scale the plot. A value 1.2 means the complete plot will have (externally) a
            radius of 1.2 instead of 1.
        :type scale: float

        :param tail: A function (typically with matplotlib settings) run after completing the plot
        :type tail: function

        :param weight_name: The Name of the edge attribute used as weight for the layer selection. (The default value is `weight`.)
        :type weight_name: str

        :returns: None

        .. _layout_reference:

        *Layout Reference*


        Layouts are lists of (edgecol, nodecol, fontcol, wid, diam, height, above, radius) tuples defining how the elements will be plotted.
        The list is processed in order and the nodes selected at each layer are selected by the column `above`. Typically, starting with a
        value below the minimum weight, all nodes enter the first layer, the second layer is defined by the nodes whose weight (in its
        entering or exiting edges) is above `above` and so on. Of course, nodes only belong to one layer which is the last layer whose
        `above` condition they meet.

        +---------+----------------+-------------------------------------------------------+
        | Column  | Description    |                                                       |
        +=========+================+=======================================================+
        | edgecol | Edge color for the layer (matplotlib compatible color constant)        |
        +---------+----------------+-------------------------------------------------------+
        | nodecol | Node color for the layer (matplotlib compatible color constant)        |
        +---------+----------------+-------------------------------------------------------+
        | fontcol | Text (name) color for the layer (matplotlib compatible color constant) |
        +---------+----------------+-------------------------------------------------------+
        | wid     | Edge width (Set units in matplotlib, possibly via head lambda)         |
        +---------+----------------+-------------------------------------------------------+
        | diam    | Node point diameter (same units)                                       |
        +---------+----------------+-------------------------------------------------------+
        | height  | Node name text height (same units)                                     |
        +---------+----------------+-------------------------------------------------------+
        | above   | Threshold to select what nodes go to each layer (increasing order)     |
        +---------+----------------+-------------------------------------------------------+
        | radius  | Radius of circle where nodes are placed (relative to total plot == 1)  |
        +---------+----------------+-------------------------------------------------------+

        """

        if head is not None:
            head()

        if scale != 1:
            g2 = nx.cycle_graph(2)
            p2 = {0: np.array([-scale, 0]), 1: np.array([scale, 0])}

            nx.draw(
                g2, p2, edge_color="#ffffff", node_color="#ffffff", node_size=1, width=0
            )

        nmw = Plot.nodes_max_weight(G)

        next_above = [above for (_, _, _, _, _, _, above, _) in layout[::-1]][:-1]
        next_below = lambda: 9e99 if len(next_above) == 0 else next_above.pop()

        for (edgecol, nodecol, fontcol, wid, diam, height, above, radius) in layout:
            edge_filter = lambda u, v: G[u][v][weight_name] > above
            node_filter = lambda u: nmw[u] > above

            g = nx.graphviews.subgraph_view(
                G, filter_node=node_filter, filter_edge=edge_filter
            )

            nx.draw(
                g,
                pos,
                edge_color=edgecol,
                width=wid,
                node_color=nodecol,
                node_size=diam,
            )

            below = next_below()

            edge_filter = (
                lambda u, v: G[u][v][weight_name] > above
                and G[u][v][weight_name] <= below
            )
            node_filter = lambda u: nmw[u] > above and nmw[u] <= below

            g = nx.graphviews.subgraph_view(
                G, filter_node=node_filter, filter_edge=edge_filter
            )

            nx.draw_networkx_labels(g, pos, font_size=height, font_color=fontcol)

        if tail is not None:
            tail()

    def layered_positions(G, layout, offs_num=2, offs_deno=3):
        """
        This returns a (networkx compatible) positioning object for plotting. This object is a dictionary

        :param G: A graph
        :type G: networkx graph

        :param layout: See layout_reference_
        :type layout: list of (edgecol, nodecol, fontcol, wid, diam, height, above, radius) tuples

        :param offs_num: (offs_num / offs_deno) Two parameters defining a fraction to offset the names and control overlapping. Otherwise,
            in all circles names would always start being positioned over the central x-axis.
        :type offs_num: int

        :param offs_deno: See previous argument.
        :type offs_num: int

        :returns: A dictionary where the keys are the node keys and the values are 2D `numpy` (x, y) vectors. This is compatible with
            networkx node positioning objects.
        :rtype: dict
        """

        nmw = Plot.nodes_max_weight(G)

        pos = {}
        for (edgecol, nodecol, fontcol, wid, diam, height, above, radius) in layout[
            ::-1
        ]:
            top = [k for k, v in nmw.items() if v > above and k not in pos]
            eps = offs_num / (offs_deno * max(len(top), 1))
            theta = 2 * np.pi * np.linspace(eps, 1 + eps, len(top) + 1)[:-1]

            pos.update(
                dict(
                    zip(
                        top,
                        np.column_stack(
                            [radius * np.cos(theta), radius * np.sin(theta)]
                        ),
                    )
                )
            )

        return pos

    def corrplot(mat, by_rows=True, color_map="Blues", precision=2):
        """
        Plots the correlation matrix using a pandas style.background_gradient.

        :param mat: A matrix whose correlations we want to plot. It can be either a numpy matrix or a pandas DataFrame.
        :type mat: matrix

        :param by_rows: Plot the correlations between the rows (default)
        :type by_rows: bool

        :param color_map: A matplotlib colormap name, such as `BuGn` used for the background See
            `here <https://matplotlib.org/users/colormaps.html>`_.
        :type color_map: string

        :param precision: Number of significant digits (ignoring leading zeroes) for numeric output.
        :type precision: int

        :returns: A correplation matrix (using pandas)
        :rtype: pandas dataframe with background_gradient

        """

        if by_rows:
            df = pd.DataFrame(np.transpose(mat))
        else:
            df = pd.DataFrame(mat)

        return (
            df.corr().style.background_gradient(cmap=color_map).set_precision(precision)
        )

