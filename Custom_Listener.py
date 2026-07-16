from collections import defaultdict, deque
from colorama import Fore, Style, init

from gen.NNGraphListener import NNGraphListener
from required_code_collection.ast import AST
from required_code_collection.make_ast_subtree import make_ast_subtree
from dimension_inference import DimensionInferencer

init(autoreset=True)


class NNGraphCustomListener(NNGraphListener):


    SUPPORTED_LAYERS = {

        "Linear": {
            "required": {
                "out_features": int
            },
            "optional": {
                "in_features": int,
                "bias": bool
            }
        },


        "Conv2d": {
            "required": {
                "out_ch": int,
                "kernel": int
            },
            "optional": {
                "in_ch": int,
                "stride": int,
                "padding": int,
                "bias": bool
            }
        },
        "Conv1d": {
                "required": {
                    "out_ch": int,
                    "kernel": int
                },
                "optional": {
                    "in_ch": int,
                    "stride": int,
                    "bias": bool
                }
        },


        "BatchNorm2d": {
            "required": {},
            "optional": {
                "num_features": int
            }
        },
        "LayerNorm": {
            "required": {},
            "optional": {
                "normalized_shape": (int, tuple)
            }
        },

        "MaxPool2d": {
            "required": {"kernel": int},
            "optional": {"stride": int}
        },

        "AvgPool2d": {
            "required": {"kernel": int},
            "optional": {"stride": int}
        },

        "Dropout": {
            "required": {"p": float},
            "optional": {}
        },

        "Flatten": {
            "required": {},
            "optional": {"start_dim": int, "end_dim": int}
        },

        "Embedding": {
            "required": {"num_embeddings": int, "embedding_dim": int},
            "optional": {}
        },

        "MultiHeadAttn": {
            "required": {
                "num_heads": int
            },
            "optional": {
                "embed_dim": int
            }
        },

        "LSTM": {
            "required": {
                "hidden_size": int
            },
            "optional": {
                "input_size": int,
                "num_layers": int
            }
        },

        "GRU": {
            "required": {
                "hidden_size": int
            },
            "optional": {
                "input_size": int
            }
        },

        # Activations
        "ReLU": {
            "required": {},
            "optional": {}
        },

        "Sigmoid": {
            "required": {},
            "optional": {}
        },

        "Tanh": {
            "required": {},
            "optional": {}
        },

        "GELU": {
            "required": {},
            "optional": {}
        },

        "Softmax": {
            "required": {"dim": int},
            "optional": {}
        },

        "LeakyReLU": {
            "required": {},
            "optional": {"negative_slope": float}
        },

        "ELU": {
            "required": {},
            "optional": {"alpha": float}
        },

        # Special operations
        "Add": {
            "required": {},
            "optional": {}
        },

        "Concat": {
            "required": {"dim": int},
            "optional": {}
        },

        "Residual": {
            "required": {},
            "optional": {}
        },

        "Split": {
            "required": {"chunks": int, "dim": int},
            "optional": {}
        }
    }

    def __init__(self):

        self.ast = AST()


        self.program_ctx = None
        self.model_ctx = None
        self.input_ctx = None
        self.output_ctx = None


        self.model_name = None
        self.input_name = None
        self.input_shape = None
        self.output_name = None

        self.nodes = {}

        self.edges = []

        self.config = {}

        self.warnings = []
        self.orphan_nodes = set()


    def _line(self, ctx):
        if ctx is not None and hasattr(ctx, "start"):
            return ctx.start.line
        return "?"

    def error(self, msg, ctx=None, hint=None):
        line = self._line(ctx)

        message = f"{Fore.RED}Error [line {line}]: {msg}{Style.RESET_ALL}"

        if hint:
            message += f"\n    {Fore.YELLOW}Hint: {hint}{Style.RESET_ALL}"

        raise ValueError(message)

    def warning(self, msg, ctx=None, hint=None):
        line = self._line(ctx)

        message = f"{Fore.YELLOW}Warning [line {line}]: {msg}{Style.RESET_ALL}"

        if hint:
            message += f"\n    {Fore.YELLOW}Hint: {hint}{Style.RESET_ALL}"

        self.warnings.append(message)

    def _print_warnings(self):
        if self.warnings:
            print(f"\n{Fore.YELLOW}===== WARNINGS ====={Style.RESET_ALL}")
            for warning in self.warnings:
                print(warning)

    # additional and helper functions
    def _parse_value_text(self, text):
        if text == "true":
            return True

        if text == "false":
            return False

        if text == "None":
            return None

        if text.startswith('"') and text.endswith('"'):
            return text[1:-1]

        if text.startswith("(") and text.endswith(")"):
            inner = text[1:-1].strip()

            if inner == "":
                return tuple()

            parts = [p.strip() for p in inner.split(",") if p.strip()]
            return tuple(int(p) for p in parts)

        if "." in text:
            return float(text)

        return int(text)

    def _dsl_type_name_from_value(self, value):
        if type(value) is int:
            return "int"
        if type(value) is float:
            return "float"
        if type(value) is bool:
            return "bool"
        if type(value) is str:
            return "string"
        if value is None:
            return "None"
        if type(value) is tuple:
            return "shape"
        return type(value).__name__

    def _expected_type_name(self, expected_type):
        if isinstance(expected_type, tuple):
            return " or ".join(self._expected_type_name(t) for t in expected_type)

        if expected_type is int:
            return "int"
        if expected_type is float:
            return "float"
        if expected_type is bool:
            return "bool"
        if expected_type is str:
            return "string"
        if expected_type is tuple:
            return "shape"

        return expected_type.__name__

    def _value_display(self, value):
        if type(value) is str:
            return f'"{value}"'
        return str(value)

    def _matches_expected_type(self, value, expected_type):
        if isinstance(expected_type, tuple):
            return any(self._matches_expected_type(value, t) for t in expected_type)

        if expected_type is int:
            return type(value) is int

        if expected_type is float:
            return type(value) is float

        if expected_type is bool:
            return type(value) is bool

        if expected_type is str:
            return type(value) is str

        if expected_type is tuple:
            return type(value) is tuple

        return isinstance(value, expected_type)

    def _declared_nodes_hint(self):
        declared = []

        if self.input_name:
            declared.append(self.input_name)

        declared.extend(self.nodes.keys())

        return "[" + ", ".join(declared) + "]"

    def _supported_layers_hint(self):
        return "[" + ", ".join(self.SUPPORTED_LAYERS.keys()) + "]"

    def _params_hint(self, node_type):
        schema = self.SUPPORTED_LAYERS[node_type]
        names = list(schema["required"].keys()) + list(schema["optional"].keys())
        return "[" + ", ".join(names) + "]"


    def exitShape(self, ctx):
        shape_text = ctx.getText()

        ctx.shape_value = tuple(
            int(x.strip()) for x in shape_text.split(",") if x.strip()
        )

        ctx.ast_value = self.ast.make_node(f"SHAPE:{shape_text}", [])
        self.ast.root = ctx.ast_value

    def exitInputDecl(self, ctx):
        input_name = ctx.ID().getText()

        self.input_name = input_name
        self.input_shape = ctx.shape().shape_value
        self.input_ctx = ctx

        children = []
        if hasattr(ctx.shape(), "ast_value"):
            children.append(ctx.shape().ast_value)

        ctx.ast_value = self.ast.make_node(f"INPUT:{input_name}", children)
        self.ast.root = ctx.ast_value


    def exitOutputDecl(self, ctx):
        output_name = ctx.ID().getText()

        self.output_name = output_name
        self.output_ctx = ctx

        ctx.ast_value = self.ast.make_node(f"OUTPUT:{output_name}", [])
        self.ast.root = ctx.ast_value


    def exitModelDecl(self, ctx):
        model_name = ctx.ID().getText()

        self.model_name = model_name
        self.model_ctx = ctx

        make_ast_subtree(
            self.ast,
            ctx,
            node_value=f"MODEL:{model_name}",
            keep_node=True
        )


    def exitLayerType(self, ctx):
        layer_name = ctx.getText()

        ctx.layer_name = layer_name

        ctx.ast_value = self.ast.make_node(f"TYPE:{layer_name}", [])
        self.ast.root = ctx.ast_value

    def exitValue(self, ctx):
        value_text = ctx.getText()

        ctx.raw_text = value_text
        ctx.value = self._parse_value_text(value_text)
        ctx.type = type(ctx.value)
        ctx.dsl_type = self._dsl_type_name_from_value(ctx.value)

        ctx.ast_value = self.ast.make_node(f"{value_text}", [])
        self.ast.root = ctx.ast_value


    def exitParam(self, ctx):
        key = ctx.ID().getText()
        value_node = ctx.value()

        ctx.param_key = key
        ctx.param_value = value_node.value
        ctx.param_type = value_node.type
        ctx.param_dsl_type = value_node.dsl_type
        ctx.param_raw_text = value_node.raw_text

        children = []
        if hasattr(ctx.value(), "ast_value"):
            children.append(ctx.value().ast_value)

        ctx.ast_value = self.ast.make_node(f"{key}", children)
        self.ast.root = ctx.ast_value

    def exitParamList(self, ctx):
        params = {}
        param_types = {}
        param_raw_texts = {}
        param_dsl_types = {}

        for param_ctx in ctx.param():
            key = param_ctx.param_key
            value = param_ctx.param_value

            if key in params:
                self.error(
                    f"Duplicate parameter '{key}' in parameter list.",
                    param_ctx
                )

            params[key] = value
            param_types[key] = param_ctx.param_type
            param_raw_texts[key] = param_ctx.param_raw_text
            param_dsl_types[key] = param_ctx.param_dsl_type

        ctx.params = params
        ctx.param_types = param_types
        ctx.param_raw_texts = param_raw_texts
        ctx.param_dsl_types = param_dsl_types

        make_ast_subtree(
            self.ast,
            ctx,
            node_value="PARAMS",
            keep_node=True
        )

    def exitNodeDecl(self, ctx):
        node_id = ctx.ID().getText()
        node_type = ctx.layerType().getText()

        # id 's should be unique
        if node_id in self.nodes:
            self.error(
                f"Node identifier '{node_id}' is already declared.",
                ctx,
                "Node identifiers must be unique within the graph block."
            )


        if node_type not in self.SUPPORTED_LAYERS:
            self.error(
                f"Unsupported layer type '{node_type}' for node '{node_id}'.",
                ctx,
                f"Supported layers are: {self._supported_layers_hint()}"
            )

        params = {}
        param_types = {}
        param_raw_texts = {}
        param_dsl_types = {}

        if ctx.paramList():
            params = ctx.paramList().params
            param_types = ctx.paramList().param_types
            param_raw_texts = ctx.paramList().param_raw_texts
            param_dsl_types = ctx.paramList().param_dsl_types


        self._check_layer_params(
            node_id=node_id,
            node_type=node_type,
            params=params,
            param_types=param_types,
            param_raw_texts=param_raw_texts,
            param_dsl_types=param_dsl_types,
            ctx=ctx
        )

        self.nodes[node_id] = {
            "type": node_type,
            "params": params,
            "param_types": param_types,
            "param_raw_texts": param_raw_texts,
            "param_dsl_types": param_dsl_types,
            "line": ctx.start.line,
            "ctx": ctx
        }

        make_ast_subtree(
            self.ast,
            ctx,
            node_value=f"{node_id}:{node_type}",
            keep_node=True
        )

    def _check_layer_params(
        self,
        node_id,
        node_type,
        params,
        param_types,
        param_raw_texts,
        param_dsl_types,
        ctx
    ):
        schema = self.SUPPORTED_LAYERS[node_type]

        required = schema["required"]
        optional = schema["optional"]

        allowed = {}
        allowed.update(required)
        allowed.update(optional)


        for param_name in required:
            if param_name not in params:
                self.error(
                    f"Missing required parameter '{param_name}' for {node_type} node '{node_id}'.",
                    ctx,
                    f"Required parameters for {node_type} are: {self._params_hint(node_type)}"
                )


        for param_name, param_value in params.items():

            if param_name not in allowed:
                self.error(
                    f"Parameter '{param_name}' is not valid for {node_type} node '{node_id}'.",
                    ctx,
                    f"Allowed parameters for {node_type} are: {self._params_hint(node_type)}"
                )

            expected_type = allowed[param_name]

            if not self._matches_expected_type(param_value, expected_type):
                expected = self._expected_type_name(expected_type)
                got = param_dsl_types[param_name]
                raw = self._value_display(param_value)

                self.error(
                    f"Parameter '{param_name}' of {node_type} expects {expected}, got {got} {raw}.",
                    ctx
                )


    def exitLabel(self, ctx):
        label_text = ctx.STRING().getText()

        ctx.ast_value = self.ast.make_node(f"LABEL:{label_text}", [])
        self.ast.root = ctx.ast_value

    def exitEdgeDecl(self, ctx):
        src = ctx.ID(0).getText()
        dst = ctx.ID(1).getText()

        label = None
        if ctx.label() is not None:
            label = ctx.label().STRING().getText()

        self.edges.append({
            "src": src,
            "dst": dst,
            "label": label,
            "line": ctx.start.line,
            "ctx": ctx
        })

        children = []
        if ctx.label() is not None and hasattr(ctx.label(), "ast_value"):
            children.append(ctx.label().ast_value)

        ctx.ast_value = self.ast.make_node(f"{src}->{dst}", children)
        self.ast.root = ctx.ast_value


    def exitGraphStatement(self, ctx):
        make_ast_subtree(
            self.ast,
            ctx,
            node_value="GRAPH_STATEMENT",
            keep_node=False
        )


    def exitGraphBlock(self, ctx):
        make_ast_subtree(
            self.ast,
            ctx,
            node_value="GRAPH",
            keep_node=True
        )


    def exitConfigStatement(self, ctx):
        key = ctx.ID().getText()
        value_text = ctx.value().getText()
        value = ctx.value().value

        self.config[key] = {
            "value": value,
            "line": ctx.start.line,
            "ctx": ctx
        }

        children = []
        if hasattr(ctx.value(), "ast_value"):
            children.append(ctx.value().ast_value)

        ctx.ast_value = self.ast.make_node(f"{key}", children)
        self.ast.root = ctx.ast_value


    def exitConfigBlock(self, ctx):
        make_ast_subtree(
            self.ast,
            ctx,
            node_value="CONFIG",
            keep_node=True
        )


    def exitProgram(self, ctx):
        self.program_ctx = ctx

        make_ast_subtree(
            self.ast,
            ctx,
            node_value="NNGraphRoot",
            keep_node=True
        )


        self.check_no_undefined_references()
        self.check_no_orphan_nodes()
        self.check_input_reachability()
        self.check_output_reachability()
        self.check_cycle_detection()
        self.check_residual_arity()

        DimensionInferencer(self).run()

        print(f"\n{Fore.CYAN}===== AST BUILT ====={Style.RESET_ALL}")
        print("Root:", self.ast.root.value)
        print("Children:", len(self.ast.root.children))

        print(f"\n{Fore.CYAN}AST Traversal:{Style.RESET_ALL}")
        print(self.ast.traverse_ast(self.ast.root))

        self._print_warnings()

        print(f"\n{Fore.CYAN}===== SEMANTIC CHECK ====={Style.RESET_ALL}")
        if self.warnings:
            print(f"{Fore.YELLOW}Semantic check completed with warnings.{Style.RESET_ALL}")
        else:
            print(f"{Fore.GREEN}Semantic check passed successfully.{Style.RESET_ALL}")



    def check_no_undefined_references(self):
        valid_nodes = set(self.nodes.keys())

        if self.input_name:
            valid_nodes.add(self.input_name)

        for edge in self.edges:
            src = edge["src"]
            dst = edge["dst"]

            if src not in valid_nodes:
                self.error(
                    f"Edge references undefined node '{src}'.",
                    edge["ctx"],
                    f"Declared nodes are: {self._declared_nodes_hint()}"
                )

            if dst not in valid_nodes:
                self.error(
                    f"Edge references undefined node '{dst}'.",
                    edge["ctx"],
                    f"Declared nodes are: {self._declared_nodes_hint()}"
                )


    def check_no_orphan_nodes(self):
        used_nodes = set()

        for edge in self.edges:
            used_nodes.add(edge["src"])
            used_nodes.add(edge["dst"])

        for node_id, node_info in self.nodes.items():


            if node_id == self.input_name or node_id == self.output_name:
                continue

            if node_id not in used_nodes:
                self.orphan_nodes.add(node_id)

                self.warning(
                    f"Node '{node_id}' is declared but never referenced in any edge.",
                    node_info["ctx"]
                )

    def check_input_reachability(self):
        if not self.input_name:
            self.error(
                "Input declaration is missing.",
                self.program_ctx
            )

        adjacency = self._build_adjacency()
        reachable = self._bfs(self.input_name, adjacency)

        for node_id, node_info in self.nodes.items():


            if node_id in self.orphan_nodes:
                continue

            if node_id not in reachable:
                self.error(
                    f"Node '{node_id}' is not reachable from input node '{self.input_name}'.",
                    node_info["ctx"],
                    "Every node must be reachable from the input node following edge directions."
                )


    def check_output_reachability(self):
        if not self.output_name:
            self.error(
                "Output declaration is missing.",
                self.program_ctx
            )

        if self.output_name not in self.nodes:
            self.error(
                f"Output node '{self.output_name}' is not declared in graph block.",
                self.output_ctx,
                f"Declared nodes are: {self._declared_nodes_hint()}"
            )

        adjacency = self._build_adjacency()
        reachable = self._bfs(self.input_name, adjacency)

        if self.output_name not in reachable:
            self.error(
                f"Output node '{self.output_name}' is not reachable from input node '{self.input_name}'.",
                self.output_ctx,
                "The output node must be reachable from at least one path from input."
            )


    def check_cycle_detection(self):
        adjacency = self._build_adjacency_with_edge_context()

        visited = set()
        visiting = set()
        stack = []

        all_nodes = set(adjacency.keys())
        for edges in adjacency.values():
            for edge in edges:
                all_nodes.add(edge["dst"])

        def dfs(node):
            visiting.add(node)
            stack.append(node)

            for edge in adjacency[node]:
                nxt = edge["dst"]

                if nxt not in visited and nxt not in visiting:
                    dfs(nxt)

                elif nxt in visiting:
                    cycle_start = stack.index(nxt)
                    cycle = stack[cycle_start:] + [nxt]

                    self.error(
                        "Cycle detected involving nodes: " + " -> ".join(cycle) + ".",
                        edge["ctx"],
                        "NNGraph graphs must be acyclic DAGs."
                    )

            visiting.remove(node)
            visited.add(node)
            stack.pop()

        for node in all_nodes:
            if node not in visited:
                dfs(node)


    def check_residual_arity(self):
        incoming_count = defaultdict(int)

        for edge in self.edges:
            incoming_count[edge["dst"]] += 1

        for node_id, node_info in self.nodes.items():
            if node_info["type"] == "Residual":
                count = incoming_count[node_id]

                if count != 2:
                    self.error(
                        f"Node '{node_id}' (Residual) has {count} incoming edges; expected exactly 2.",
                        node_info["ctx"]
                    )


    def _build_adjacency(self):
        adjacency = defaultdict(list)

        valid_nodes = set(self.nodes.keys())

        if self.input_name:
            valid_nodes.add(self.input_name)

        for edge in self.edges:
            src = edge["src"]
            dst = edge["dst"]

            if src in valid_nodes and dst in valid_nodes:
                adjacency[src].append(dst)

        return adjacency

    def _build_adjacency_with_edge_context(self):
        adjacency = defaultdict(list)

        valid_nodes = set(self.nodes.keys())

        if self.input_name:
            valid_nodes.add(self.input_name)

        for edge in self.edges:
            src = edge["src"]
            dst = edge["dst"]

            if src in valid_nodes and dst in valid_nodes:
                adjacency[src].append(edge)

        return adjacency

    def _bfs(self, start, adjacency):
        visited = set()
        queue = deque([start])

        while queue:
            node = queue.popleft()

            if node in visited:
                continue

            visited.add(node)

            for nxt in adjacency[node]:
                if nxt not in visited:
                    queue.append(nxt)

        return visited