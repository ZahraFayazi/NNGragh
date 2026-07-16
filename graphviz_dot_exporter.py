import os


class GraphvizDotExporter:


    SPECIAL_OPS = {"Add", "Concat", "Residual", "Split"}
    ACTIVATIONS = {
        "ReLU",
        "Sigmoid",
        "Tanh",
        "GELU",
        "Softmax",
        "LeakyReLU",
        "ELU",
    }

    def __init__(self, listener, include_inferred_params=True):
        self.listener = listener
        self.model_name = listener.model_name or "NNGraph"
        self.input_name = listener.input_name
        self.input_shape = listener.input_shape
        self.output_name = listener.output_name
        self.nodes = listener.nodes
        self.edges = listener.edges
        self.include_inferred_params = include_inferred_params



    def generate(self):
        lines = [
            f"digraph {self._quote(self.model_name)} {{",
            "    graph [",
            "        rankdir=LR,",
            "        labelloc=t,",
            f"        label={self._quote('NNGraph model: ' + self.model_name)},",
            '        fontname="Helvetica",',
            "        fontsize=18,",
            "        nodesep=0.55,",
            "        ranksep=0.80,",
            "        splines=polyline",
            "    ];",
            '    node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#F7F7F7"];',
            '    edge [fontname="Helvetica", fontsize=10, color="#555555"];',
            "",
        ]

        lines.append(self._emit_input_node())

        for node_id, node_info in self.nodes.items():
            lines.append(self._emit_graph_node(node_id, node_info))

        if self.edges:
            lines.append("")

        for edge in self.edges:
            lines.append(self._emit_edge(edge))

        lines.append("}")
        return "\n".join(lines) + "\n"

    def write(self, path):
        output_dir = os.path.dirname(os.path.abspath(path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        dot_source = self.generate()

        with open(path, "w", encoding="utf-8") as file:
            file.write(dot_source)

        return path


    def _emit_input_node(self):
        shape_text = self._format_shape(self.input_shape)
        label = f"{self.input_name}\nInput tensor({shape_text})"

        attrs = {
            "label": label,
            "shape": "oval",
            "fillcolor": "#D9EAF7",
            "color": "#2878B5",
            "penwidth": 1.6,
        }


        if self.input_name == self.output_name:
            attrs["peripheries"] = 2
            attrs["color"] = "#2E8B57"

        return f"    {self._quote(self.input_name)} {self._attrs(attrs)};"

    def _emit_graph_node(self, node_id, node_info):
        node_type = node_info["type"]
        params_text = self._format_params(node_info)
        call_text = f"{node_type}({params_text})" if params_text else f"{node_type}()"
        label = f"{node_id}\n{call_text}"

        attrs = {
            "label": label,
            "shape": "box",
            "fillcolor": "#F7F7F7",
            "color": "#666666",
        }

        if node_type in self.SPECIAL_OPS:
            attrs.update(
                shape="diamond",
                fillcolor="#FFF2CC",
                color="#B8860B",
            )
        elif node_type in self.ACTIVATIONS:
            attrs.update(
                shape="ellipse",
                fillcolor="#E8F5E9",
                color="#2E7D32",
            )

        if node_id == self.output_name:
            attrs.update(
                peripheries=2,
                color="#2E8B57",
                penwidth=1.8,
            )

        return f"    {self._quote(node_id)} {self._attrs(attrs)};"



    def _emit_edge(self, edge):
        attrs = {}
        label = self._clean_edge_label(edge.get("label"))

        if label:
            attrs["label"] = label

            lowered = label.lower()
            if any(token in lowered for token in ("residual", "shortcut", "skip")):
                attrs["style"] = "dashed"
                attrs["color"] = "#C0392B"
                attrs["fontcolor"] = "#C0392B"
                attrs["penwidth"] = 1.5

        attr_text = f" {self._attrs(attrs)}" if attrs else ""

        return (
            f"    {self._quote(edge['src'])} -> "
            f"{self._quote(edge['dst'])}{attr_text};"
        )



    def _format_params(self, node_info):
        params = node_info.get("params", {}) or {}
        raw_texts = node_info.get("param_raw_texts", {}) or {}
        inferred = node_info.get("inferred_params", {}) or {}

        parts = []

        for key, raw_value in raw_texts.items():
            parts.append(f"{key}={raw_value}")

        if self.include_inferred_params:

            for key, value in inferred.items():
                if key in raw_texts:
                    continue
                parts.append(f"{key}={self._format_value(value)} [inferred]")


        if not parts:
            for key, value in params.items():
                parts.append(f"{key}={self._format_value(value)}")

        return ", ".join(parts)

    @staticmethod
    def _format_shape(shape):
        if not shape:
            return ""
        return ", ".join(str(value) for value in shape)

    @classmethod
    def _format_value(cls, value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "None"
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(value, tuple):
            return "(" + ", ".join(cls._format_value(item) for item in value) + ")"
        return str(value)

    @staticmethod
    def _clean_edge_label(label):
        if label is None:
            return ""

        label = str(label).strip()

        if len(label) >= 2 and label[0] == '"' and label[-1] == '"':
            label = label[1:-1]

        return label

    @classmethod
    def _attrs(cls, attrs):
        parts = []

        for key, value in attrs.items():
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            elif isinstance(value, (int, float)):
                rendered = str(value)
            else:
                rendered = cls._quote(value)

            parts.append(f"{key}={rendered}")

        return "[" + ", ".join(parts) + "]"

    @staticmethod
    def _quote(value):
        text = str(value)
        text = text.replace("\\", "\\\\")
        text = text.replace('"', '\\"')
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\n", "\\n")
        return f'"{text}"'
