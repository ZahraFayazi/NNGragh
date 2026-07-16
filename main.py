from antlr4 import *
import argparse
import os
import subprocess
import traceback

from gen.NNGraphLexer import NNGraphLexer
from gen.NNGraphParser import NNGraphParser
from Custom_Listener import NNGraphCustomListener

from required_code_collection.ast_to_networkx_graph import show_ast
from pytorch_code_generator import PyTorchCodeGenerator
from graphviz_dot_exporter import GraphvizDotExporter


def main(arguments):



    try:
        stream = FileStream(arguments.input, encoding="utf8")

    except Exception as e:
        print("\nInput file error.")
        print(type(e).__name__ + ":", e)
        return 1



    lexer = NNGraphLexer(stream)
    tokens = CommonTokenStream(lexer)

    parser = NNGraphParser(tokens)
    parse_tree = parser.program()

    if parser.getNumberOfSyntaxErrors() > 0:
        print("\nSyntax error found.")
        print("Semantic analysis, graph export, and code generation skipped.")
        return 1



    listener = NNGraphCustomListener()
    walker = ParseTreeWalker()

    print(
        "\n--- Starting Parsing, AST Construction, "
        f"and Semantic Analysis on {arguments.input} ---"
    )

    try:
        walker.walk(listener, parse_tree)

    except ValueError as e:
        print("\nSemantic analysis failed.")
        print(e)
        print(
            "\nGraph export and code generation skipped "
            "until semantic errors are fixed."
        )
        return 1

    except Exception as e:
        print("\nUnexpected error during parsing / semantic analysis.")
        print(type(e).__name__ + ":", e)

        if arguments.debug:
            traceback.print_exc()

        return 1

    print("\nSemantic analysis completed successfully.")
    print("Parsed config:", listener.config)



    if (
        arguments.dot_output
        or arguments.png_output
        or arguments.print_dot
    ):
        print("\n--- Exporting NNGraph to Graphviz ---")

        try:
            dot_exporter = GraphvizDotExporter(listener)
            dot_source = dot_exporter.generate()

            # DOT path used for both DOT export and PNG generation
            dot_path = arguments.dot_output or "output/model.dot"

            # Create DOT output directory
            dot_directory = os.path.dirname(
                os.path.abspath(dot_path)
            )

            if dot_directory:
                os.makedirs(dot_directory, exist_ok=True)


            dot_exporter.write(dot_path)

            print("Graphviz DOT exported successfully.")
            print(f"DOT output file: {dot_path}")

            # Generate or overwrite model.png
            if arguments.png_output:
                png_directory = os.path.dirname(
                    os.path.abspath(arguments.png_output)
                )

                if png_directory:
                    os.makedirs(png_directory, exist_ok=True)

                subprocess.run(
                    [
                        "dot",
                        "-Tpng",
                        dot_path,
                        "-o",
                        arguments.png_output,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

                print("Graphviz PNG generated successfully.")
                print(f"PNG output file: {arguments.png_output}")

            if arguments.print_dot:
                print("\n--- Generated Graphviz DOT ---")
                print(dot_source)

        except FileNotFoundError:
            print("\nGraphviz executable 'dot' was not found.")
            print(
                "Install Graphviz and make sure its bin directory "
                "is added to the system PATH."
            )
            return 1

        except subprocess.CalledProcessError as e:
            print("\nGraphviz failed to generate the PNG image.")

            if e.stderr:
                print(e.stderr)

            if arguments.debug:
                traceback.print_exc()

            return 1

        except Exception as e:
            print("\nGraphviz DOT/PNG export failed.")
            print(type(e).__name__ + ":", e)

            if arguments.debug:
                traceback.print_exc()

            return 1



    print("\n--- Generating PyTorch Code ---")

    try:
        generator = PyTorchCodeGenerator(listener)

        include_main = True if arguments.include_main else None

        generated_code = generator.generate(
            include_main=include_main
        )


        compile(
            generated_code,
            arguments.output,
            "exec"
        )

        output_directory = os.path.dirname(
            os.path.abspath(arguments.output)
        )

        if output_directory:
            os.makedirs(output_directory, exist_ok=True)


        with open(arguments.output, "w", encoding="utf8") as file:
            file.write(generated_code)

        print("PyTorch code generated successfully.")
        print(f"Output file: {arguments.output}")

        if arguments.print_code:
            print("\n--- Generated PyTorch Code ---")
            print(generated_code)

    except ValueError as e:
        print("\nCode generation failed.")
        print(e)
        return 1

    except SyntaxError as e:
        print("\nGenerated code has Python syntax error.")
        print(type(e).__name__ + ":", e)

        if arguments.debug:
            traceback.print_exc()

        return 1

    except Exception as e:
        print("\nUnexpected error during code generation.")
        print(type(e).__name__ + ":", e)

        if arguments.debug:
            traceback.print_exc()

        return 1


    if arguments.show_ast:
        print("\n--- Showing AST Graph ---")

        try:
            if listener.ast.root is not None:
                show_ast(listener.ast.root)
                print("AST graph display completed.")

            else:
                print("AST root is None.")

        except Exception as e:
            print("\nAST graph display failed.")
            print(type(e).__name__ + ":", e)

            if arguments.debug:
                traceback.print_exc()

    print("\nCompilation completed successfully.")
    return 0


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description=(
            "NNGraph DSL Compiler: parse, semantic analysis, "
            "AST construction, Graphviz export, "
            "and PyTorch code generation."
        )
    )

    argparser.add_argument(
        "-i",
        "--input",
        help="Input NNGraph source file",
        default="input/test.txt"
    )

    argparser.add_argument(
        "-o",
        "--output",
        help="Output generated PyTorch Python file",
        default="output/generated_model.py"
    )

    argparser.add_argument(
        "--include-main",
        action="store_true",
        help="Force generating if __name__ == '__main__' block"
    )

    argparser.add_argument(
        "--print-code",
        action="store_true",
        help="Print generated PyTorch code"
    )

    argparser.add_argument(
        "--dot-output",
        help="Export the validated user graph as a Graphviz DOT file",
        default="output/model.dot"
    )

    argparser.add_argument(
        "--png-output",
        help="Generate the Graphviz graph as a PNG image",
        default="output/model.png"
    )

    argparser.add_argument(
        "--print-dot",
        action="store_true",
        help="Print generated Graphviz DOT source"
    )

    argparser.add_argument(
        "--show-ast",
        action="store_true",
        help="Show AST graph"
    )

    argparser.add_argument(
        "--debug",
        action="store_true",
        help="Print full traceback"
    )

    args = argparser.parse_args()

    raise SystemExit(main(args))