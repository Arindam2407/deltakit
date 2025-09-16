# (c) Copyright Riverlane 2020-2025.
import pytest
from deltakit_core.decoding_graphs import (
    DecodingEdge,
    DecodingHyperEdge,
    DecodingHyperGraph,
    DecodingHyperMultiGraph,
    EdgeRecord,
    NXDecodingGraph,
    OrderedDecodingEdges,
)
from deltakit_core.decoding_graphs._decoding_graph_tools import (
    has_contiguous_nodes,
    hypergraph_to_weighted_edge_list,
    inverse_logical_at_boundary,
    is_logical_along_boundary,
    is_single_connected_component,
    single_boundary_is_last_node,
    worst_case_num_detectors,
)

# TODO
# Add tests for a code with no coordinates


def decoding_hypergraph_with_hyperedges():
    edge_ints = [(0, 1, 2), (2, 3, 6), (1, 4, 5), (6, 5, 4), (0, 6, 5)]
    edge_data = [
        (DecodingHyperEdge(vertices), EdgeRecord(p_err=0.1)) for vertices in edge_ints
    ]
    return DecodingHyperGraph(edge_data)


def decoding_hypergraph_without_hyperedges():
    edge_ints = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (2, 3),
        (3, 6),
        (2, 5),
        (1, 4),
        (6, 5),
        (5, 4),
    ]
    edge_data = [
        (DecodingHyperEdge(vertices), EdgeRecord(p_err=0.1)) for vertices in edge_ints
    ]
    return DecodingHyperGraph(edge_data)


@pytest.mark.parametrize(
    "hypergraph, expected_repr",
    [
        (
            decoding_hypergraph_with_hyperedges(),
            {
                (frozenset({0, 1, 2}), 0.1),
                (frozenset({2, 3, 6}), 0.1),
                (frozenset({1, 4, 5}), 0.1),
                (frozenset({4, 5, 6}), 0.1),
                (frozenset({0, 5, 6}), 0.1),
            },
        ),
        (
            decoding_hypergraph_without_hyperedges(),
            {
                (frozenset({0, 1}), 0.1),
                (frozenset({0, 2}), 0.1),
                (frozenset({0, 3}), 0.1),
                (frozenset({1, 2}), 0.1),
                (frozenset({2, 3}), 0.1),
                (frozenset({3, 6}), 0.1),
                (frozenset({2, 5}), 0.1),
                (frozenset({1, 4}), 0.1),
                (frozenset({6, 5}), 0.1),
                (frozenset({5, 4}), 0.1),
            },
        ),
    ],
)
def test_integer_hypergraph_repr_is_expected(hypergraph, expected_repr):
    assert set(hypergraph_to_weighted_edge_list(hypergraph)) == expected_repr


@pytest.mark.parametrize(
    "graph, start_index",
    [
        (DecodingHyperGraph(DecodingHyperEdge((i, i + 1)) for i in range(0, 10, 2)), 0),
        (DecodingHyperGraph([DecodingHyperEdge((2, 1)), DecodingHyperEdge((1, 0))]), 0),
        (DecodingHyperGraph(DecodingHyperEdge((i, i + 1)) for i in range(1, 11, 2)), 1),
        (NXDecodingGraph.from_edge_list([(0, 1), (1, 2), (2, 3)]), 0),
        (NXDecodingGraph.from_edge_list([(3, 4), (5, 6), (7, 8)]), 3),
    ],
)
def test_decoding_hyper_graph_with_contiguous_nodes_has_contiguous_nodes(
    graph, start_index
):
    assert has_contiguous_nodes(graph, start_index)


@pytest.mark.parametrize(
    "graph",
    [
        DecodingHyperGraph(DecodingHyperEdge((i, i + 1)) for i in range(0, 10, 2)),
        NXDecodingGraph.from_edge_list((i, i + 1) for i in range(0, 4, 2)),
    ],
)
def test_decoding_hyper_graph_with_contiguous_nodes_but_wrong_start_index_doesnt_have_contiguous_nodes(
    graph,
):
    assert not has_contiguous_nodes(graph, 1)


@pytest.mark.parametrize(
    "graph",
    [
        DecodingHyperGraph([DecodingHyperEdge((0, 2))]),
        DecodingHyperGraph([DecodingHyperEdge((0, 1)), DecodingHyperEdge((3, 4))]),
        NXDecodingGraph.from_edge_list([(1, 2), (3, 0), (5, 6)]),
    ],
)
def test_decoding_hyper_graph_without_contiguous_nodes_doesnt_have_contiguous_nodes(
    graph,
):
    assert not has_contiguous_nodes(graph)


def test_decoding_graph_with_single_boundary_as_last_node():
    graph = NXDecodingGraph.from_edge_list([(0, 1), (2, 3)], boundaries=(3,))
    assert single_boundary_is_last_node(graph)


@pytest.mark.parametrize(
    "graph",
    [
        NXDecodingGraph.from_edge_list([(0, 1), (2, 3)], boundaries=(3, 1)),
        NXDecodingGraph.from_edge_list([(0, 1), (2, 3)], boundaries=()),
    ],
)
def test_decoding_graph_with_wrong_number_of_boundaries_is_not_single_boundary(graph):
    assert not single_boundary_is_last_node(graph)


def test_decoding_graph_with_boundary_not_last_node_is_not_last_node():
    graph = NXDecodingGraph.from_edge_list([(0, 1), (2, 3)], boundaries=(1,))
    assert not single_boundary_is_last_node(graph)


def test_decoding_graph_with_two_connected_components_is_not_single_connected_component():
    graph = NXDecodingGraph.from_edge_list([(0, 1), (2, 3)])
    assert not is_single_connected_component(graph)


def test_decoding_graph_with_single_connected_component_is_single_connected_component():
    graph = NXDecodingGraph.from_edge_list([(0, 1), (1, 2), (2, 3)])
    assert is_single_connected_component(graph)


def test_decoding_graph_connected_only_by_the_boundaries_is_not_single_connected_component():
    graph = NXDecodingGraph.from_edge_list(
        [(0, 1), (1, 2), (2, 3), (3, 4)], boundaries=[2]
    )
    assert not is_single_connected_component(graph)


def decoding_graph_logicals_no_boundaries():
    return (
        NXDecodingGraph.from_edge_list(
            [
                (DecodingEdge(i % 10, (i + 3) % 10), EdgeRecord(p_err=0.5))
                for i in range(21)
            ]
        ),
        [frozenset([DecodingEdge(1, 4)]), frozenset([DecodingEdge(2, 5)])],
    )


def decoding_graph_no_logicals_boundaries_16():
    return NXDecodingGraph.from_edge_list(
        [(DecodingEdge(i, (i + 1) % 17), EdgeRecord(p_err=0.5)) for i in range(17)],
        boundaries=[16],
    ), []


def decoding_graph_logicals_boundaries_20():
    return (
        NXDecodingGraph.from_edge_list(
            [(DecodingEdge(i, (i + 1) % 21), EdgeRecord(p_err=0.5)) for i in range(21)],
            boundaries=[20],
        ),
        [frozenset([DecodingEdge(1, 2), DecodingEdge(3, 4), DecodingEdge(5, 6)])],
    )


def decoding_graph_and_logical_weighted():
    edges = [
        (DecodingEdge(0, 1), EdgeRecord(p_err=0.1)),
        (DecodingEdge(0, 2), EdgeRecord(p_err=0.3)),
        (DecodingEdge(1, 2), EdgeRecord(p_err=0.5)),
    ]
    return NXDecodingGraph.from_edge_list(edges), []


def decoding_graph_and_logicals_hyper_multi():
    graph = DecodingHyperMultiGraph(
        [
            ((l0 := DecodingHyperEdge([0, 1, 2])), EdgeRecord(0.01)),
            (DecodingHyperEdge([0]), EdgeRecord(0.02)),
            (DecodingHyperEdge([2]), EdgeRecord(0.01)),
            (DecodingHyperEdge([0, 1, 2]), EdgeRecord(0.01)),
            (DecodingHyperEdge([2]), EdgeRecord(0.03)),
            ((l1 := DecodingHyperEdge([1, 2])), EdgeRecord(0.01)),
        ]
    )
    return graph, [frozenset([(l0, 0), (l1, 0)])]


class TestInverseLogical:
    @pytest.fixture(
        scope="class",
        params=[
            NXDecodingGraph.from_edge_list(
                [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (4, 3)], boundaries=[4]
            ),
            NXDecodingGraph.from_edge_list(
                [(i, (i + 1) % 21) for i in range(21)], boundaries=[20]
            ),
        ],
    )
    def graph(self, request) -> NXDecodingGraph:
        return request.param

    @staticmethod
    def get_all_edges_to_boundaries(decoding_graph: NXDecodingGraph):
        all_edges_to_boundaries = set()
        for boundary in decoding_graph.boundaries:
            all_edges_to_boundaries.update(decoding_graph.incident_edges(boundary))
        return all_edges_to_boundaries

    def test_inverse_logical_returned_on_single_boundary_graph(self):
        graph = NXDecodingGraph.from_edge_list(
            [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (4, 3)], boundaries=[0]
        )
        logical = OrderedDecodingEdges.from_syndrome_indices([(0, 1), (0, 3)])
        expected = OrderedDecodingEdges.from_syndrome_indices([(0, 2), (0, 4)])
        assert inverse_logical_at_boundary(graph, logical) == set(expected)

    def test_inverse_logical_returned_on_multi_boundary_graph(self):
        graph = NXDecodingGraph.from_edge_list(
            [(0, 1), (0, 2), (0, 3), (3, 6), (0, 4), (1, 2), (2, 3), (4, 3)],
            boundaries=[0, 3],
        )
        logical = OrderedDecodingEdges.from_syndrome_indices([(0, 1), (0, 3), (2, 3)])
        expected = OrderedDecodingEdges.from_syndrome_indices(
            [(0, 2), (0, 4), (4, 3), (3, 6)]
        )
        assert inverse_logical_at_boundary(graph, logical) == set(expected)

    def test_inverse_logical_returned_with_empty_input_logical(self, graph):
        expected = TestInverseLogical.get_all_edges_to_boundaries(graph)
        assert inverse_logical_at_boundary(graph, set()) == set(expected)

    def test_no_inverse_logical_returned_with_input_logical_is_complete(self, graph):
        logical = TestInverseLogical.get_all_edges_to_boundaries(graph)
        assert inverse_logical_at_boundary(graph, logical) == set()

    def test_inverse_logical_raises_exception_when_logical_not_along_boundary(self):
        graph = NXDecodingGraph.from_edge_list(
            [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (4, 3)], boundaries=[0]
        )
        logical = OrderedDecodingEdges.from_syndrome_indices([(0, 1), (4, 3)])
        with pytest.raises(
            ValueError,
            match="The logical given was not entirely along "
            "the boundary, so the inverse logical is undefined.",
        ):
            inverse_logical_at_boundary(graph, logical)

    def test_inverse_logical_raises_exception_when_logical_not_on_graph(self):
        graph = NXDecodingGraph.from_edge_list(
            [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (4, 3)], boundaries=[0]
        )
        logical = OrderedDecodingEdges.from_syndrome_indices([(2, 3), (10, 9)])
        with pytest.raises(
            ValueError,
            match="The logical given was not entirely along "
            "the boundary, so the inverse logical is undefined.",
        ):
            inverse_logical_at_boundary(graph, logical)

    def test_inverse_logical_raises_exception_when_no_boundary_in_graph(self):
        graph = NXDecodingGraph.from_edge_list(
            [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (4, 3)], boundaries=[]
        )
        logical = OrderedDecodingEdges.from_syndrome_indices([(2, 3), (10, 9)])
        with pytest.raises(
            ValueError,
            match="The logical given was not entirely along "
            "the boundary, so the inverse logical is undefined.",
        ):
            inverse_logical_at_boundary(graph, logical)


def decoding_graph_boundaries_16():
    return NXDecodingGraph.from_edge_list(
        [(i, (i + 1) % 17) for i in range(17)], boundaries=[16]
    )


def test_logicals_can_be_determined_to_be_entirely_along_boundary():
    dec_graph = decoding_graph_boundaries_16()
    logicals = [{dec_graph.edges[16]}]
    assert is_logical_along_boundary(dec_graph, logicals)


def test_logicals_can_be_determined_to_not_be_along_boundary():
    dec_graph = decoding_graph_boundaries_16()
    logicals = [{dec_graph.edges[10]}]
    assert not is_logical_along_boundary(dec_graph, logicals)


def test_logicals_can_be_determined_to_not_be_along_entire_boundary_if_only_a_subset_of_a_logical_touches_boundary():
    dec_graph = decoding_graph_boundaries_16()
    logicals = [{dec_graph.edges[10], dec_graph.edges[12]}]
    assert not is_logical_along_boundary(dec_graph, logicals)


def test_worst_case_num_detectors_returns_num_detectors_for_all_p_err_1_graph():
    graph = DecodingHyperGraph(
        [
            (DecodingHyperEdge((0,)), EdgeRecord(p_err=1)),
            (DecodingHyperEdge((0, 1)), EdgeRecord(p_err=1)),
            (DecodingHyperEdge((2, 3)), EdgeRecord(p_err=1)),
            (DecodingHyperEdge((1, 3, 4)), EdgeRecord(p_err=1)),
            (DecodingHyperEdge((4,)), EdgeRecord(p_err=1)),
        ]
    )
    assert worst_case_num_detectors(graph, 1e-9) == len(graph.nodes) - len(
        graph.boundaries
    )


def test_worst_case_num_detectors_returns_0_for_all_p_err_0_graph():
    graph = DecodingHyperGraph(
        [
            (DecodingHyperEdge((0,)), EdgeRecord(p_err=0)),
            (DecodingHyperEdge((0, 2, 1, 9)), EdgeRecord(p_err=0)),
            (DecodingHyperEdge((2,)), EdgeRecord(p_err=0)),
            (DecodingHyperEdge((1, 3, 4)), EdgeRecord(p_err=0)),
            (DecodingHyperEdge((4, 9)), EdgeRecord(p_err=0)),
        ]
    )
    assert worst_case_num_detectors(graph, 1e-9) == 0


def test_worst_case_num_detectors_returns_num_detectors_for_0_target_logical_error():
    graph = DecodingHyperGraph(
        [
            (DecodingHyperEdge((0,)), EdgeRecord(p_err=0.3)),
            (DecodingHyperEdge((0, 1)), EdgeRecord(p_err=0.01)),
            (DecodingHyperEdge((2, 3)), EdgeRecord(p_err=0.03)),
            (DecodingHyperEdge((1, 3, 4)), EdgeRecord(p_err=0.1)),
            (DecodingHyperEdge((4,)), EdgeRecord(p_err=0.25)),
        ]
    )
    assert worst_case_num_detectors(graph, 0) == len(graph.nodes) - len(
        graph.boundaries
    )


def test_worst_case_num_detectors_returns_0_for_1_target_logical_error():
    graph = DecodingHyperGraph(
        [
            (DecodingHyperEdge((0,)), EdgeRecord(p_err=0.3)),
            (DecodingHyperEdge((0, 1)), EdgeRecord(p_err=0.01)),
            (DecodingHyperEdge((2, 3)), EdgeRecord(p_err=0.03)),
            (DecodingHyperEdge((1, 3, 4)), EdgeRecord(p_err=0.1)),
            (DecodingHyperEdge((4,)), EdgeRecord(p_err=0.25)),
        ]
    )
    assert worst_case_num_detectors(graph, 1) == 0
