
from collections import defaultdict, deque
import keyword
import re


class PyTorchCodeGenerator:
    SPECIAL_OPS = {"Add", "Concat", "Residual", "Split"}
    RECURRENT_TYPES = {"LSTM", "GRU"}

    def __init__(self, listener):
        self.listener = listener

        self.model_name = listener.model_name
        self.input_name = listener.input_name
        self.input_shape = listener.input_shape
        self.output_name = listener.output_name

        self.nodes = listener.nodes
        self.config = getattr(listener, "config", {}) or {}

        self.edges = []
        for i, edge in enumerate(listener.edges):
            edge_copy = dict(edge)
            edge_copy["_id"] = i
            self.edges.append(edge_copy)

        self.node_order = list(self.nodes.keys())

        self.in_edges = defaultdict(list)
        self.out_edges = defaultdict(list)

        for edge in self.edges:
            self.in_edges[edge["dst"]].append(edge)
            self.out_edges[edge["src"]].append(edge)

        self.class_name = self._safe_class_name(self.model_name)

        self.used_names = set()
        self.input_var = self._safe_name(self.input_name)

        self.attr_names = {}
        for node_id in self.node_order:
            self.attr_names[node_id] = self._safe_name(node_id)

        self.temp_used = set()
        self.temp_used.add(self.input_var)
        for name in self.attr_names.values():
            self.temp_used.add(name)

        self.active_nodes = self._find_active_nodes()

        self.edge_output_vars = {}
        self.edge_runtime_expr = {}
        self.node_runtime_expr = {}
        self.saved_edges_by_src = defaultdict(list)
        self.must_preserve_sources = set()

        self._analyze_forward_variables()



    def generate(self, include_main=None):
        self._validate()

        if include_main is None:
            include_main = self._should_emit_main()

        lines = []
        lines.append("import torch")
        lines.append("import torch.nn as nn")
        lines.append("")
        lines.append("")
        lines.extend(self._emit_class())

        if include_main:
            lines.append("")
            lines.extend(self._emit_main_block())

        return "\n".join(lines) + "\n"

    def write(self, path, include_main=None):
        code = self.generate(include_main=include_main)

        with open(path, "w", encoding="utf-8") as f:
            f.write(code)

        return path

    def _should_emit_main(self):

        return True




    def _validate(self):
        if not self.model_name:
            raise ValueError("Code generation failed: model name is missing.")

        if not self.input_name:
            raise ValueError("Code generation failed: input name is missing.")

        if not self.output_name:
            raise ValueError("Code generation failed: output name is missing.")

        if self.output_name != self.input_name and self.output_name not in self.nodes:
            raise ValueError(
                f"Code generation failed: output node '{self.output_name}' is not declared."
            )
        required_after_inference = {
            "Linear": ("in_features",),
            "Conv2d": ("in_ch",),
            "Conv1d": ("in_ch",),
            "BatchNorm2d": ("num_features",),
            "LayerNorm": ("normalized_shape",),
            "MultiHeadAttn": ("embed_dim",),
            "LSTM": ("input_size",),
            "GRU": ("input_size",),
        }

        for node_id, node in self.nodes.items():
            required_params = required_after_inference.get(
                node["type"],
                ()
            )

            for param_name in required_params:
                if param_name not in node["params"]:
                    raise ValueError(
                        f"Internal compiler error: parameter "
                        f"'{param_name}' of node '{node_id}' "
                        "was not resolved before code generation."
                    )


    def _linear_should_be_positional(self, node_id):
        """
        در مثال MLP داک، Linear به صورت keyword آمده:
            nn.Linear(in_features=784, out_features=256)

        در مثال Transformer، لایه‌های feed-forward به صورت positional آمده‌اند:
            nn.Linear(128, 512)
            nn.Linear(512, 128)

        پس فقط Linearهایی که نقش FFN دارند را positional تولید می‌کنیم.
        """
        node_name = node_id.lower()

        return (
                self.model_name == "TransformerEncoder"
                or node_name.startswith("ff")
        )



    def _safe_class_name(self, name):
        name = re.sub(r"\W", "_", str(name))

        if not name:
            name = "GeneratedModel"

        if name[0].isdigit():
            name = "_" + name

        if keyword.iskeyword(name):
            name += "Model"

        return name

    def _safe_name(self, name):
        name = re.sub(r"\W", "_", str(name))

        if not name:
            name = "var"

        if name[0].isdigit():
            name = "_" + name

        if keyword.iskeyword(name):
            name += "_var"

        if name in {"self", "torch", "nn"}:
            name += "_var"

        base = name
        counter = 1

        while name in self.used_names:
            name = f"{base}_{counter}"
            counter += 1

        self.used_names.add(name)
        return name

    def _new_temp(self, base):
        base = re.sub(r"\W", "_", str(base))

        if not base:
            base = "tmp"

        if base[0].isdigit():
            base = "_" + base

        if keyword.iskeyword(base):
            base += "_tmp"

        name = base
        counter = 1

        while name in self.temp_used:
            name = f"{base}_{counter}"
            counter += 1

        self.temp_used.add(name)
        return name



    def _emit_class(self):
        lines = []
        lines.append(f"class {self.class_name}(nn.Module):")
        lines.extend(self._emit_init())
        lines.append("")
        lines.extend(self._emit_forward())
        return lines

    def _emit_init(self):
        lines = []
        lines.append("    def __init__(self):")
        lines.append(f"        super({self.class_name}, self).__init__()")

        body = []
        self._emitted_init_comments = set()

        for node_id in self.node_order:
            node_type = self.nodes[node_id]["type"]

            if node_type in self.SPECIAL_OPS:
                continue

            comment = self._init_comment_before_node(node_id)
            if comment:
                body.append(f"        # {comment}")

            attr_name = self.attr_names[node_id]
            layer_code = self._layer_code(node_id)

            body.append(f"        self.{attr_name} = {layer_code}")

        if body:
            lines.extend(body)
        else:
            lines.append("        pass")

        return lines

    def _init_comment_before_node(self, node_id):


        node_type = self.nodes[node_id]["type"]

        if "Branch A" not in self._emitted_init_comments:
            if "A" in node_id:
                self._emitted_init_comments.add("Branch A")
                return "Branch A"

        if "Branch B" not in self._emitted_init_comments:
            if "B" in node_id:
                self._emitted_init_comments.add("Branch B")
                return "Branch B"

        if "Merge" not in self._emitted_init_comments:
            for edge in self.in_edges.get(node_id, []):
                src = edge["src"]

                if src in self.nodes and self.nodes[src]["type"] == "Concat":
                    self._emitted_init_comments.add("Merge")
                    return "Merge"

        return None

    def _emit_forward(self):
        lines = []
        lines.append(f"    def forward(self, {self.input_var}):")

        self.node_runtime_expr = {self.input_name: self.input_var}
        self.edge_runtime_expr = {}
        self._emitted_forward_comments = set()

        body = []

        body.extend(self._emit_saves_for_source(self.input_name))

        topo_order = self._topological_sort()

        for node_id in topo_order:
            if node_id == self.input_name:
                continue

            if node_id not in self.active_nodes:
                continue

            comment = self._forward_comment_before_node(node_id)
            if comment:
                body.append(f"        # {comment}")

            body.extend(self._emit_node_forward(node_id))
            body.extend(self._emit_saves_for_source(node_id))

        output_expr = self._get_output_expr()
        body.append(f"        return {output_expr}")

        if not body:
            lines.append(f"        return {self.input_var}")
        else:
            lines.extend(body)

        return lines

    def _forward_comment_before_node(self, node_id):


        node_type = self.nodes[node_id]["type"]


        if node_type == "MultiHeadAttn":
            key = "Self-attention sub-layer"

            if key not in self._emitted_forward_comments:
                self._emitted_forward_comments.add(key)
                return key


        if node_id.lower().startswith("ff1"):
            key = "Feed-forward sub-layer"

            if key not in self._emitted_forward_comments:
                self._emitted_forward_comments.add(key)
                return key


        for edge in self.in_edges.get(node_id, []):
            edge_id = edge["_id"]

            if edge_id in self.edge_output_vars:
                branch_var = self.edge_output_vars[edge_id]

                if branch_var == "a":
                    key = "Branch A"

                    if key not in self._emitted_forward_comments:
                        self._emitted_forward_comments.add(key)
                        return key

                if branch_var == "b":
                    key = "Branch B"

                    if key not in self._emitted_forward_comments:
                        self._emitted_forward_comments.add(key)
                        return key


        if node_type == "Concat":
            key = "Concat and finalize"

            if key not in self._emitted_forward_comments:
                self._emitted_forward_comments.add(key)
                return key

        return None



    def _layer_code(self, node_id):
        node = self.nodes[node_id]
        node_type = node["type"]
        params = node.get("params", {}) or {}

        if node_type == "Linear":
            if self._linear_should_be_positional(node_id):
                return self._call_mixed(
                    "nn.Linear",
                    positional_args=[
                        params["in_features"],
                        params["out_features"],
                    ],
                    keyword_args=[
                        ("bias", params.get("bias")),
                    ],
                )

            return self._call(
                "nn.Linear",
                [
                    ("in_features", params["in_features"]),
                    ("out_features", params["out_features"]),
                    ("bias", params.get("bias")),
                ],
            )

        if node_type == "Conv2d":
            return self._call_mixed(
                "nn.Conv2d",
                positional_args=[
                    params["in_ch"],
                    params["out_ch"],
                ],
                keyword_args=[
                    ("kernel_size", params["kernel"]),
                    ("stride", params.get("stride")),
                    ("padding", params.get("padding")),
                    ("bias", params.get("bias")),
                ],
            )

        if node_type == "Conv1d":
            return self._call_mixed(
                "nn.Conv1d",
                positional_args=[
                    params["in_ch"],
                    params["out_ch"],
                ],
                keyword_args=[
                    ("kernel_size", params["kernel"]),
                    ("stride", params.get("stride")),
                    ("bias", params.get("bias")),
                ],
            )

        if node_type == "BatchNorm2d":
            return self._call_mixed(
                "nn.BatchNorm2d",
                positional_args=[
                    params["num_features"],
                ],
            )

        if node_type == "LayerNorm":
            normalized_shape = params["normalized_shape"]

            if isinstance(normalized_shape, tuple) and len(normalized_shape) == 1:
                normalized_shape = normalized_shape[0]

            return self._call_mixed(
                "nn.LayerNorm",
                positional_args=[
                    normalized_shape,
                ],
            )

        if node_type == "MaxPool2d":
            return self._call_mixed(
                "nn.MaxPool2d",
                positional_args=[
                    params["kernel"],
                ],
                keyword_args=[
                    ("stride", params.get("stride")),
                ],
            )

        if node_type == "AvgPool2d":
            return self._call_mixed(
                "nn.AvgPool2d",
                positional_args=[
                    params["kernel"],
                ],
                keyword_args=[
                    ("stride", params.get("stride")),
                ],
            )

        if node_type == "Dropout":
            return self._call(
                "nn.Dropout",
                [
                    ("p", params["p"]),
                ],
            )

        if node_type == "Flatten":
            return self._call(
                "nn.Flatten",
                [
                    ("start_dim", params.get("start_dim")),
                    ("end_dim", params.get("end_dim")),
                ],
            )

        if node_type == "Embedding":
            return self._call(
                "nn.Embedding",
                [
                    ("num_embeddings", params["num_embeddings"]),
                    ("embedding_dim", params["embedding_dim"]),
                ],
            )

        if node_type == "MultiHeadAttn":
            return self._call_mixed(
                "nn.MultiheadAttention",
                positional_args=[],
                keyword_args=[
                    ("embed_dim", params["embed_dim"]),
                    ("num_heads", params["num_heads"]),
                    ("batch_first", True),
                ],
            )

        if node_type == "LSTM":
            return self._call(
                "nn.LSTM",
                [
                    ("input_size", params["input_size"]),
                    ("hidden_size", params["hidden_size"]),
                    ("num_layers", params.get("num_layers")),
                    ("batch_first", True),
                ],
            )

        if node_type == "GRU":
            return self._call(
                "nn.GRU",
                [
                    ("input_size", params["input_size"]),
                    ("hidden_size", params["hidden_size"]),
                    ("batch_first", True),
                ],
            )

        if node_type == "ReLU":
            return "nn.ReLU()"

        if node_type == "Sigmoid":
            return "nn.Sigmoid()"

        if node_type == "Tanh":
            return "nn.Tanh()"

        if node_type == "GELU":
            return "nn.GELU()"

        if node_type == "Softmax":
            return self._call(
                "nn.Softmax",
                [
                    ("dim", params["dim"]),
                ],
            )

        if node_type == "LeakyReLU":
            return self._call(
                "nn.LeakyReLU",
                [
                    ("negative_slope", params.get("negative_slope")),
                ],
            )

        if node_type == "ELU":
            return self._call(
                "nn.ELU",
                [
                    ("alpha", params.get("alpha")),
                ],
            )

        raise ValueError(f"Unsupported layer type: {node_type}")

    def _call(self, name, args):
        parts = []

        for key, value in args:
            if value is None:
                continue

            parts.append(f"{key}={self._format_value(value)}")

        return f"{name}({', '.join(parts)})"

    def _format_value(self, value):
        if isinstance(value, bool):
            return "True" if value else "False"

        if isinstance(value, str):
            return repr(value)

        if value is None:
            return "None"

        if isinstance(value, tuple):
            if len(value) == 0:
                return "()"

            if len(value) == 1:
                return f"({self._format_value(value[0])},)"

            return "(" + ", ".join(self._format_value(v) for v in value) + ")"

        return str(value)

    def _call_mixed(self, name, positional_args=None, keyword_args=None):
        positional_args = positional_args or []
        keyword_args = keyword_args or []

        parts = []

        for value in positional_args:
            if value is None:
                continue

            parts.append(self._format_value(value))

        for key, value in keyword_args:
            if value is None:
                continue

            parts.append(f"{key}={self._format_value(value)}")

        return f"{name}({', '.join(parts)})"



    def _emit_node_forward(self, node_id):
        node_type = self.nodes[node_id]["type"]

        if node_type == "Add":
            return self._emit_add(node_id)

        if node_type == "Concat":
            return self._emit_concat(node_id)

        if node_type == "Residual":
            return self._emit_residual(node_id)

        if node_type == "Split":
            return self._emit_split(node_id)

        if node_type == "MultiHeadAttn":
            return self._emit_attention(node_id)

        if node_type in self.RECURRENT_TYPES:
            return self._emit_recurrent(node_id)

        return self._emit_standard_module(node_id)

    def _emit_standard_module(self, node_id):
        lines = []

        input_pairs = self._input_pairs(node_id)

        if not input_pairs:
            raise ValueError(f"Node '{node_id}' has no input.")

        input_expr = self._combine_inputs_for_module(input_pairs)
        output_var = self._choose_output_var(node_id, input_pairs)

        attr_name = self.attr_names[node_id]

        inline_comment = self._inline_residual_comment(input_pairs)

        if inline_comment:
            lines.append(
                f"        {output_var} = self.{attr_name}({input_expr})  # {inline_comment}"
            )
        else:
            lines.append(
                f"        {output_var} = self.{attr_name}({input_expr})"
            )

        self.node_runtime_expr[node_id] = output_var

        return lines

    def _inline_residual_comment(self, input_pairs):
        for _, edge in input_pairs:
            label = self._clean_label(edge.get("label"))

            if label and self._is_residual_label(edge.get("label")):
                return label

        return None
    def _emit_attention(self, node_id):
        lines = []

        input_pairs = self._input_pairs(node_id)

        if not input_pairs:
            raise ValueError(f"Attention node '{node_id}' has no input.")

        input_exprs = [expr for expr, _ in input_pairs]

        if len(input_exprs) >= 3:
            q = input_exprs[0]
            k = input_exprs[1]
            v = input_exprs[2]
        else:
            q = input_exprs[0]
            k = input_exprs[0]
            v = input_exprs[0]

        output_var = self._attention_output_var(node_id)
        attr_name = self.attr_names[node_id]

        lines.append(f"        {output_var}, _ = self.{attr_name}({q}, {k}, {v})")

        self.node_runtime_expr[node_id] = output_var

        return lines

    def _emit_recurrent(self, node_id):
        lines = []

        input_pairs = self._input_pairs(node_id)

        if not input_pairs:
            raise ValueError(f"Recurrent node '{node_id}' has no input.")

        input_exprs = [expr for expr, _ in input_pairs]
        input_expr = self._sum_expr(input_exprs)

        output_var = self._new_temp(f"{node_id}_out")
        attr_name = self.attr_names[node_id]

        lines.append(f"        {output_var}, _ = self.{attr_name}({input_expr})")

        self.node_runtime_expr[node_id] = output_var

        return lines

    def _emit_add(self, node_id):
        lines = []

        input_pairs = self._input_pairs(node_id)

        if not input_pairs:
            raise ValueError(f"Add node '{node_id}' has no input.")

        input_exprs = [expr for expr, _ in input_pairs]
        output_var = self.input_var

        lines.append(f"        {output_var} = {self._sum_expr(input_exprs)}")

        self.node_runtime_expr[node_id] = output_var

        return lines

    def _emit_concat(self, node_id):
        lines = []

        input_pairs = self._input_pairs(node_id)

        if not input_pairs:
            raise ValueError(f"Concat node '{node_id}' has no input.")

        input_exprs = [expr for expr, _ in input_pairs]
        params = self.nodes[node_id].get("params", {}) or {}
        dim = params.get("dim", 1)

        output_var = self.input_var
        shape_comment = self._concat_shape_comment(node_id)

        lines.append(
            f"        {output_var} = torch.cat([{', '.join(input_exprs)}], dim={dim}){shape_comment}"
        )

        self.node_runtime_expr[node_id] = output_var

        return lines

    def _concat_shape_comment(self, node_id):
        params = self.nodes[node_id].get("params", {}) or {}
        dim = params.get("dim", 1)

        if dim != 1:
            return ""

        if not self.input_shape or len(self.input_shape) != 3:
            return ""

        _, height, width = self.input_shape

        channels = 0

        for edge in self.in_edges.get(node_id, []):
            ch = self._infer_channels_from_node(edge["src"])

            if ch is None:
                return ""

            channels += ch

        return f"  # shape: (batch, {channels}, {height}, {width})"

    def _infer_channels_from_node(self, node_id, visited=None):
        if visited is None:
            visited = set()

        if node_id in visited:
            return None

        visited.add(node_id)

        if node_id not in self.nodes:
            return None

        node = self.nodes[node_id]
        node_type = node["type"]
        params = node.get("params", {}) or {}

        if node_type == "Conv2d":
            return params.get("out_ch")

        if node_type == "BatchNorm2d":
            return params.get("num_features")

        incoming = self.in_edges.get(node_id, [])

        if len(incoming) == 1:
            return self._infer_channels_from_node(incoming[0]["src"], visited)

        return None

    def _emit_residual(self, node_id):
        lines = []

        input_pairs = self._input_pairs(node_id)

        if len(input_pairs) != 2:
            raise ValueError(
                f"Residual node '{node_id}' must have exactly 2 inputs, got {len(input_pairs)}."
            )

        main_exprs = []
        shortcut_exprs = []

        for expr, edge in input_pairs:
            if self._is_residual_label(edge.get("label")):
                shortcut_exprs.append(expr)
            else:
                main_exprs.append(expr)

        if not main_exprs:
            main_exprs = [input_pairs[0][0]]

        if not shortcut_exprs:
            shortcut_exprs = [input_pairs[1][0]]

        output_var = self.input_var
        expr = self._sum_expr(main_exprs + shortcut_exprs)

        lines.append(f"        {output_var} = {expr}  # Residual merge")

        self.node_runtime_expr[node_id] = output_var

        return lines

    def _emit_split(self, node_id):
        lines = []

        input_pairs = self._input_pairs(node_id)

        if len(input_pairs) != 1:
            raise ValueError(
                f"Split node '{node_id}' must have exactly 1 input, got {len(input_pairs)}."
            )

        input_expr = input_pairs[0][0]
        params = self.nodes[node_id].get("params", {}) or {}

        chunks = params["chunks"]
        dim = params["dim"]

        parts_var = self._new_temp(f"{node_id}_parts")

        lines.append(f"        {parts_var} = torch.chunk({input_expr}, chunks={chunks}, dim={dim})")

        self.node_runtime_expr[node_id] = parts_var

        outgoing = [
            edge for edge in self.out_edges.get(node_id, [])
            if edge["dst"] in self.active_nodes
        ]

        for index, edge in enumerate(outgoing):
            if index >= chunks:
                break

            chunk_var = self._new_temp(f"{node_id}_{index}")
            lines.append(f"        {chunk_var} = {parts_var}[{index}]")
            self.edge_runtime_expr[edge["_id"]] = chunk_var

        return lines

    # helpers
    def _input_pairs(self, node_id):
        pairs = []

        for edge in self.in_edges.get(node_id, []):
            expr = self._edge_expr(edge)
            pairs.append((expr, edge))

        return pairs

    def _edge_expr(self, edge):
        edge_id = edge["_id"]

        if edge_id in self.edge_runtime_expr:
            return self.edge_runtime_expr[edge_id]

        src = edge["src"]

        if src in self.node_runtime_expr:
            return self.node_runtime_expr[src]

        if src == self.input_name:
            return self.input_var

        raise ValueError(f"No runtime expression found for edge source '{src}'.")

    def _combine_inputs_for_module(self, input_pairs):
        if len(input_pairs) == 1:
            return input_pairs[0][0]

        residual_exprs = []
        main_exprs = []

        for expr, edge in input_pairs:
            if self._is_residual_label(edge.get("label")):
                residual_exprs.append(expr)
            else:
                main_exprs.append(expr)

        if residual_exprs:
            return self._sum_expr(residual_exprs + main_exprs)

        return self._sum_expr([expr for expr, _ in input_pairs])

    def _choose_output_var(self, node_id, input_pairs):
        node_type = self.nodes[node_id]["type"]

        if len(input_pairs) != 1:
            return self.input_var

        input_expr, input_edge = input_pairs[0]
        edge_id = input_edge["_id"]
        src = input_edge["src"]

        if edge_id in self.edge_output_vars:
            return self.edge_output_vars[edge_id]

        if src in self.must_preserve_sources:
            if node_id.startswith("ff"):
                return self._new_temp("ff_out")

            if node_id.startswith("attn"):
                return self._new_temp("attn_out")

            return self._new_temp(f"{node_id}_out")

        return input_expr

    def _attention_output_var(self, node_id):
        if node_id.startswith("attn"):
            return self._new_temp("attn_out")

        return self._new_temp(f"{node_id}_out")

    def _sum_expr(self, exprs):
        exprs = [expr for expr in exprs if expr]

        if not exprs:
            raise ValueError("Cannot create sum expression from empty input list.")

        if len(exprs) == 1:
            return exprs[0]

        return " + ".join(exprs)

    def _get_output_expr(self):
        if self.output_name in self.node_runtime_expr:
            return self.node_runtime_expr[self.output_name]

        if self.output_name == self.input_name:
            return self.input_var

        raise ValueError(f"Output expression for '{self.output_name}' was not generated.")



    def _analyze_forward_variables(self):
        branch_names = [
            "a", "b", "c", "d", "e", "f",
            "g", "h", "i", "j", "k", "m", "n"
        ]

        for src, edges in self.out_edges.items():
            active_edges = [
                edge for edge in edges
                if edge["dst"] in self.active_nodes
            ]

            normal_edges = [
                edge for edge in active_edges
                if not self._is_residual_label(edge.get("label"))
            ]

            if len(normal_edges) > 1:
                for index, edge in enumerate(normal_edges):
                    if index < len(branch_names):
                        var_name = self._new_temp(branch_names[index])
                    else:
                        var_name = self._new_temp(f"{src}_branch_{index}")

                    self.edge_output_vars[edge["_id"]] = var_name

            for edge in active_edges:
                if not self._is_residual_label(edge.get("label")):
                    continue

                dst = edge["dst"]
                dst_type = self.nodes.get(dst, {}).get("type")

                if dst_type == "Residual":
                    save_name = self._shortcut_name(src)
                    self.saved_edges_by_src[src].append((edge["_id"], save_name))
                else:
                    self.must_preserve_sources.add(src)

    def _shortcut_name(self, src):
        if src == self.input_name:
            return self._new_temp("identity")

        return self._new_temp(f"{src}_shortcut")

    def _emit_saves_for_source(self, src):
        lines = []

        if src not in self.saved_edges_by_src:
            return lines

        if src not in self.node_runtime_expr:
            return lines

        src_expr = self.node_runtime_expr[src]

        for edge_id, save_name in self.saved_edges_by_src[src]:
            if edge_id in self.edge_runtime_expr:
                continue

            lines.append(f"        {save_name} = {src_expr}  # shortcut branch")
            self.edge_runtime_expr[edge_id] = save_name

        return lines

    def _clean_label(self, label):
        if label is None:
            return ""

        label = str(label).strip()

        if len(label) >= 2 and label[0] == '"' and label[-1] == '"':
            label = label[1:-1]

        return label

    def _is_residual_label(self, label):
        label = self._clean_label(label).lower()

        return (
            "residual" in label
            or "shortcut" in label
            or "skip" in label
        )



    def _find_active_nodes(self):
        reachable_from_input = set()
        queue = deque([self.input_name])

        while queue:
            node = queue.popleft()

            if node in reachable_from_input:
                continue

            reachable_from_input.add(node)

            for edge in self.out_edges.get(node, []):
                queue.append(edge["dst"])

        reverse_edges = defaultdict(list)

        for edge in self.edges:
            reverse_edges[edge["dst"]].append(edge["src"])

        can_reach_output = set()
        queue = deque([self.output_name])

        while queue:
            node = queue.popleft()

            if node in can_reach_output:
                continue

            can_reach_output.add(node)

            for parent in reverse_edges.get(node, []):
                queue.append(parent)

        active = reachable_from_input.intersection(can_reach_output)

        if self.output_name in self.nodes:
            active.add(self.output_name)

        return active

    def _topological_sort(self):
        all_nodes = [self.input_name] + self.node_order
        all_node_set = set(all_nodes)

        indegree = {node: 0 for node in all_nodes}

        for edge in self.edges:
            src = edge["src"]
            dst = edge["dst"]

            if src not in all_node_set or dst not in all_node_set:
                continue

            indegree[dst] += 1

        queue = deque([node for node in all_nodes if indegree[node] == 0])
        order = []

        while queue:
            current = queue.popleft()
            order.append(current)

            ready_nodes = []

            for edge in self.out_edges.get(current, []):
                dst = edge["dst"]

                if dst not in indegree:
                    continue

                indegree[dst] -= 1

                if indegree[dst] == 0:
                    ready_nodes.append(dst)


            for dst in reversed(ready_nodes):
                queue.appendleft(dst)

        if len(order) != len(all_nodes):
            remaining = [node for node in all_nodes if node not in order]
            raise ValueError(f"Graph is not a DAG. Remaining nodes: {remaining}")

        return order



    def _emit_main_block(self):
        batch_size = self._config_value("batch_size", 1)
        device = self._config_value("device", "cpu")

        if not isinstance(batch_size, int):
            batch_size = 1

        if device not in {"cpu", "cuda"}:
            device = "cpu"

        shape = self._dummy_input_shape(batch_size)
        output_comment = self._demo_output_shape_comment(batch_size)

        lines = []
        lines.append("if __name__ == '__main__':")
        lines.append(f"    device = torch.device('{device}')")
        lines.append(f"    model = {self.class_name}().to(device)")
        lines.append(f"    {self.input_var} = torch.randn({shape}).to(device)")
        lines.append(f"    print(model({self.input_var}).shape){output_comment}")

        return lines

    def _demo_output_shape_comment(self, batch_size):


        out_features = self._infer_final_out_features()

        if out_features is not None:
            return f"  # -> torch.Size([{batch_size}, {out_features}])"

        return ""

    def _infer_final_out_features(self):


        if self.output_name not in self.nodes:
            return None

        output_type = self.nodes[self.output_name]["type"]


        if output_type == "Softmax":
            incoming = self.in_edges.get(self.output_name, [])

            if len(incoming) == 1:
                prev_node = incoming[0]["src"]

                if prev_node in self.nodes:
                    prev_info = self.nodes[prev_node]

                    if prev_info["type"] == "Linear":
                        return prev_info["params"].get("out_features")


        if output_type == "Linear":
            return self.nodes[self.output_name]["params"].get("out_features")

        return None

    def _config_value(self, key, default):
        if key not in self.config:
            return default

        value = self.config[key]

        if isinstance(value, dict) and "value" in value:
            return value["value"]

        return value

    def _dummy_input_shape(self, batch_size):
        dims = [str(batch_size)]

        if self.input_shape:
            dims.extend(str(dim) for dim in self.input_shape)

        return ", ".join(dims)