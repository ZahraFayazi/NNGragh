from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TensorMeta:


    family: str = "unknown"
    width: Optional[int] = None
    rank: Optional[int] = None
    axis: Optional[int] = None

    @property
    def known(self):
        return self.family != "unknown" and self.width is not None


UNKNOWN = TensorMeta()


class DimensionInferencer:

    PASS_THROUGH = {
        "ReLU",
        "Sigmoid",
        "Tanh",
        "GELU",
        "Softmax",
        "LeakyReLU",
        "ELU",
        "Dropout",
    }

    INFERABLE_PARAMS = {
        "Linear": ("in_features",),
        "Conv2d": ("in_ch",),
        "Conv1d": ("in_ch",),
        "BatchNorm2d": ("num_features",),
        "LayerNorm": ("normalized_shape",),
        "MultiHeadAttn": ("embed_dim",),
        "LSTM": ("input_size",),
        "GRU": ("input_size",),
    }

    def __init__(self, listener):
        self.listener = listener
        self.nodes = listener.nodes
        self.edges = listener.edges
        self.input_name = listener.input_name

        self.in_edges = defaultdict(list)
        self.out_edges = defaultdict(list)

        for edge in self.edges:
            self.in_edges[edge["dst"]].append(edge)
            self.out_edges[edge["src"]].append(edge)

        self.meta = {
            self.input_name: UNKNOWN
        }

    def run(self):
        for node_id in self._topological_order():

            if node_id == self.input_name:
                continue

            if node_id not in self.nodes:
                continue

            incoming_meta = [
                self.meta.get(edge["src"], UNKNOWN)
                for edge in self.in_edges.get(node_id, [])
            ]

            self.meta[node_id] = self._infer_node(
                node_id,
                incoming_meta
            )

        self._validate_all_conditionally_required_params()

    def _topological_order(self):
        vertices = [self.input_name] + list(self.nodes.keys())
        vertex_set = set(vertices)

        indegree = {
            vertex: 0
            for vertex in vertices
        }

        adjacency = defaultdict(list)

        for edge in self.edges:
            src = edge["src"]
            dst = edge["dst"]

            if src not in vertex_set or dst not in vertex_set:
                continue

            adjacency[src].append(dst)
            indegree[dst] += 1

        queue = deque(
            vertex
            for vertex in vertices
            if indegree[vertex] == 0
        )

        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            for nxt in adjacency[current]:
                indegree[nxt] -= 1

                if indegree[nxt] == 0:
                    queue.append(nxt)

        return result

    def _infer_node(self, node_id, incoming):
        node = self.nodes[node_id]
        node_type = node["type"]
        params = node["params"]
        ctx = node.get("ctx")


        if node_type in ("Add", "Residual"):
            return self._merge_equal_inputs(node_id, incoming)


        if node_type == "Concat":
            return self._concat_meta(
                node_id,
                incoming,
                params["dim"]
            )


        common = self._merge_equal_inputs(node_id, incoming)

        if node_type == "Linear":
            inferred = (
                common.width
                if common.family == "feature"
                else None
            )

            self._set_or_validate(
                node_id,
                "in_features",
                inferred
            )

            return TensorMeta(
                family="feature",
                width=params["out_features"],
                rank=common.rank,
                axis=self._last_axis(common.rank)
            )

        if node_type == "Conv2d":
            inferred = (
                common.width
                if common.family == "conv2d"
                else None
            )

            self._set_or_validate(
                node_id,
                "in_ch",
                inferred
            )

            return TensorMeta(
                family="conv2d",
                width=params["out_ch"],
                rank=4,
                axis=1
            )

        if node_type == "Conv1d":
            inferred = (
                common.width
                if common.family == "conv1d"
                else None
            )

            self._set_or_validate(
                node_id,
                "in_ch",
                inferred
            )

            return TensorMeta(
                family="conv1d",
                width=params["out_ch"],
                rank=3,
                axis=1
            )

        if node_type == "BatchNorm2d":
            inferred = (
                common.width
                if common.family == "conv2d"
                else None
            )

            self._set_or_validate(
                node_id,
                "num_features",
                inferred
            )

            num_features = params.get("num_features")

            if num_features is None:
                return UNKNOWN

            return TensorMeta(
                family="conv2d",
                width=num_features,
                rank=4,
                axis=1
            )

        if node_type == "LayerNorm":
            inferred = (
                common.width
                if common.family == "feature"
                else None
            )

            self._set_or_validate(
                node_id,
                "normalized_shape",
                inferred,
                equivalent=self._layer_norm_equivalent
            )

            width = self._last_normalized_dim(
                params.get("normalized_shape")
            )

            if width is None:
                return UNKNOWN

            return TensorMeta(
                family="feature",
                width=width,
                rank=common.rank,
                axis=self._last_axis(common.rank)
            )

        if node_type == "MultiHeadAttn":
            inferred = (
                common.width
                if common.family == "feature"
                else None
            )

            self._set_or_validate(
                node_id,
                "embed_dim",
                inferred
            )

            embed_dim = params.get("embed_dim")
            num_heads = params.get("num_heads")

            if (
                    embed_dim is not None
                    and num_heads is not None
                    and embed_dim % num_heads != 0
            ):
                self.listener.error(
                    f"Parameter 'embed_dim' ({embed_dim}) of "
                    f"MultiHeadAttn node '{node_id}' must be "
                    f"divisible by num_heads ({num_heads}).",
                    ctx
                )

            if embed_dim is None:
                return UNKNOWN

            return TensorMeta(
                family="feature",
                width=embed_dim,
                rank=3,
                axis=2
            )

        if node_type in ("LSTM", "GRU"):
            inferred = (
                common.width
                if common.family == "feature"
                else None
            )

            self._set_or_validate(
                node_id,
                "input_size",
                inferred
            )

            return TensorMeta(
                family="feature",
                width=params["hidden_size"],
                rank=3,
                axis=2
            )

        if node_type == "Embedding":
            return TensorMeta(
                family="feature",
                width=params["embedding_dim"],
                rank=3,
                axis=2
            )

        if node_type in self.PASS_THROUGH:
            return common

        if node_type in ("MaxPool2d", "AvgPool2d"):
            if common.family == "conv2d":
                return common

            return UNKNOWN

        if node_type == "Flatten":
            return self._flatten_meta(
                common,
                params
            )

        if node_type == "Split":
            chunks = params["chunks"]

            if chunks <= 0:
                self.listener.error(
                    f"Parameter 'chunks' of Split node "
                    f"'{node_id}' must be greater than zero.",
                    ctx
                )

            return self._split_meta(
                common,
                chunks,
                params["dim"]
            )

        return UNKNOWN

    def _set_or_validate(
            self,
            node_id,
            param_name,
            inferred,
            equivalent=None
    ):
        if inferred is None:
            return

        node = self.nodes[node_id]
        params = node["params"]
        ctx = node.get("ctx")


        if param_name not in params:
            params[param_name] = inferred

            node.setdefault(
                "inferred_params",
                {}
            )[param_name] = inferred

            return


        actual = params[param_name]

        if equivalent is not None:
            same = equivalent(actual, inferred)
        else:
            same = actual == inferred

        if not same:
            self.listener.error(
                f"Parameter '{param_name}' of "
                f"{node['type']} node '{node_id}' is {actual}, "
                f"but graph inference requires {inferred}.",
                ctx,
                "Remove the parameter to let the compiler infer it, "
                "or correct its value."
            )

    def _validate_all_conditionally_required_params(self):
        for node_id, node in self.nodes.items():

            inferable = self.INFERABLE_PARAMS.get(
                node["type"],
                ()
            )

            for param_name in inferable:

                if param_name in node["params"]:
                    continue

                self.listener.error(
                    f"Cannot infer parameter '{param_name}' for "
                    f"{node['type']} node '{node_id}'.",
                    node.get("ctx"),
                    "Specify it explicitly because no preceding "
                    "declared node provides enough dimension information."
                )

    def _merge_equal_inputs(self, node_id, incoming):
        if not incoming:
            return UNKNOWN


        if any(not meta.known for meta in incoming):
            return UNKNOWN

        first = incoming[0]

        for other in incoming[1:]:

            if self._same_dimension(first, other):
                continue

            node = self.nodes[node_id]

            self.listener.error(
                f"Incoming tensor dimensions of node "
                f"'{node_id}' are incompatible.",
                node.get("ctx"),
                f"Known incoming dimensions are: {incoming}"
            )

        return first

    def _concat_meta(self, node_id, incoming, dim):
        if not incoming:
            return UNKNOWN

        if any(not meta.known for meta in incoming):
            return UNKNOWN

        first = incoming[0]


        for other in incoming[1:]:

            if (
                    other.family != first.family
                    or other.rank != first.rank
                    or other.axis != first.axis
            ):
                return UNKNOWN


        if self._is_target_axis(first, dim):
            return TensorMeta(
                family=first.family,
                width=sum(meta.width for meta in incoming),
                rank=first.rank,
                axis=first.axis
            )


        if all(
                meta.width == first.width
                for meta in incoming[1:]
        ):
            return first

        node = self.nodes[node_id]

        self.listener.error(
            f"Concat node '{node_id}' joins another axis, "
            "but its incoming tracked dimensions do not match.",
            node.get("ctx")
        )

    def _split_meta(self, meta, chunks, dim):
        if not meta.known:
            return UNKNOWN


        if self._is_target_axis(meta, dim):

            if meta.width % chunks != 0:
                return UNKNOWN

            return TensorMeta(
                family=meta.family,
                width=meta.width // chunks,
                rank=meta.rank,
                axis=meta.axis
            )


        if meta.rank is not None:
            return meta

        return UNKNOWN

    def _flatten_meta(self, meta, params):
        if not meta.known or meta.rank is None:
            return UNKNOWN

        rank = meta.rank

        start = params.get("start_dim", 1)
        end = params.get("end_dim", -1)

        start = self._normalize_axis(start, rank)
        end = self._normalize_axis(end, rank)

        if start is None or end is None or start > end:
            return UNKNOWN

        axis = self._normalize_axis(meta.axis, rank)

        if axis is None:
            return UNKNOWN

        removed_dimensions = end - start
        new_rank = rank - removed_dimensions


        if axis < start:
            return TensorMeta(
                family=meta.family,
                width=meta.width,
                rank=new_rank,
                axis=axis
            )


        if axis > end:
            return TensorMeta(
                family=meta.family,
                width=meta.width,
                rank=new_rank,
                axis=axis - removed_dimensions
            )


        if start == end == axis:
            return TensorMeta(
                family="feature",
                width=meta.width,
                rank=new_rank,
                axis=start
            )


        return UNKNOWN

    @staticmethod
    def _same_dimension(first, second):
        return (
            first.family,
            first.width,
            first.rank,
            first.axis
        ) == (
            second.family,
            second.width,
            second.rank,
            second.axis
        )

    @staticmethod
    def _last_axis(rank):
        if rank is None:
            return -1

        return rank - 1

    def _is_target_axis(self, meta, dim):
        if meta.axis is None:
            return False

        if meta.rank is None:
            return dim == -1 and meta.axis == -1

        return (
                self._normalize_axis(dim, meta.rank)
                == self._normalize_axis(meta.axis, meta.rank)
        )

    @staticmethod
    def _normalize_axis(axis, rank):
        if axis is None or rank is None:
            return None

        normalized = (
            axis + rank
            if axis < 0
            else axis
        )

        if 0 <= normalized < rank:
            return normalized

        return None

    @staticmethod
    def _last_normalized_dim(value):
        if type(value) is int:
            return value

        if type(value) is tuple and value:
            return value[-1]

        return None

    @classmethod
    def _layer_norm_equivalent(cls, actual, inferred):
        return cls._last_normalized_dim(actual) == inferred

