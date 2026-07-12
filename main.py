from antlr4 import *
import argparse
import os
import traceback

from gen.NNGraphLexer import NNGraphLexer
from gen.NNGraphParser import NNGraphParser
from Custom_Listener import NNGraphCustomListener

from required_code_collection.ast_to_networkx_graph import show_ast
from pytorch_code_generator import PyTorchCodeGenerator


def main(arguments):
    # ===============================
    # 1. Read input source
    # ===============================
    try:
        stream = FileStream(arguments.input, encoding="utf8")
    except Exception as e:
        print("\nInput file error.")
        print(type(e).__name__ + ":", e)
        return 1

    # ===============================
    # 2. Lexer + Parser
    # ===============================
    lexer = NNGraphLexer(stream)
    tokens = CommonTokenStream(lexer)

    parser = NNGraphParser(tokens)
    parse_tree = parser.program()

    if parser.getNumberOfSyntaxErrors() > 0:
        print("\nSyntax error found.")
        print("Semantic analysis and code generation skipped.")
        return 1

    # ===============================
    # 3. AST + Semantic Analysis
    # ===============================
    listener = NNGraphCustomListener()
    walker = ParseTreeWalker()

    print(f"\n--- Starting Parsing, AST Construction, and Semantic Analysis on {arguments.input} ---")

    try:
        walker.walk(listener, parse_tree)

    except ValueError as e:
        print("\nSemantic analysis failed.")
        print(e)
        print("\nCode generation skipped until semantic errors are fixed.")
        return 1

    except Exception as e:
        print("\nUnexpected error during parsing / semantic analysis.")
        print(type(e).__name__ + ":", e)

        if arguments.debug:
            traceback.print_exc()

        return 1

    print("\nSemantic analysis completed successfully.")

    # برای چک کردن اینکه config واقعاً parse شده یا نه
    print("Parsed config:", listener.config)

    # ===============================
    # 4. PyTorch Code Generation
    # ===============================
    print("\n--- Generating PyTorch Code ---")

    try:
        generator = PyTorchCodeGenerator(listener)

        # نکته مهم:
        # اگر --include-main زده شده باشد، main اجباری تولید می‌شود.
        # اگر زده نشده باشد، None می‌دهیم تا generator خودش تصمیم بگیرد.
        # یعنی اگر config وجود داشته باشد، main تولید می‌شود.
        include_main = True if arguments.include_main else None

        generated_code = generator.generate(
            include_main=include_main
        )

        compile(generated_code, arguments.output, "exec")

        output_dir = os.path.dirname(os.path.abspath(arguments.output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(arguments.output, "w", encoding="utf8") as f:
            f.write(generated_code)

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

    # ===============================
    # 5. Optional AST Display
    # ===============================
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
        description="NNGraph DSL Compiler: parse, semantic analysis, AST construction, and PyTorch code generation."
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
        default="generated_model.py"
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



# from antlr4 import *
# import argparse
# import os
# import traceback
#
# from gen.NNGraphLexer import NNGraphLexer
# from gen.NNGraphParser import NNGraphParser
# from Custom_Listener import NNGraphCustomListener
#
# from required_code_collection.ast_to_networkx_graph import show_ast
# from pytorch_code_generator import PyTorchCodeGenerator
#
#
# def main(arguments):
#     # ===============================
#     # 1. Read input source
#     # ===============================
#     try:
#         stream = FileStream(arguments.input, encoding="utf8")
#     except Exception as e:
#         print("\nInput file error.")
#         print(type(e).__name__ + ":", e)
#         return 1
#
#     # ===============================
#     # 2. Lexer + Parser
#     # ===============================
#     lexer = NNGraphLexer(stream)
#     tokens = CommonTokenStream(lexer)
#
#     parser = NNGraphParser(tokens)
#     parse_tree = parser.program()
#
#     # اگر syntax error وجود داشته باشد، semantic و code generation اجرا نشود
#     if parser.getNumberOfSyntaxErrors() > 0:
#         print("\nSyntax error found.")
#         print("Semantic analysis, AST display, and code generation skipped.")
#         return 1
#
#     # ===============================
#     # 3. AST + Semantic Analysis
#     # ===============================
#     listener = NNGraphCustomListener()
#     walker = ParseTreeWalker()
#
#     print(f"\n--- Starting Parsing, AST Construction, and Semantic Analysis on {arguments.input} ---")
#
#     try:
#         walker.walk(listener, parse_tree)
#
#     except ValueError as e:
#         print("\nSemantic analysis failed.")
#         print(e)
#         print("\nAST graph display skipped.")
#         print("Code generation skipped until semantic errors are fixed.")
#         return 1
#
#     except Exception as e:
#         print("\nUnexpected error during parsing / semantic analysis.")
#         print(type(e).__name__ + ":", e)
#
#         if arguments.debug:
#             traceback.print_exc()
#
#         print("\nCode generation skipped.")
#         return 1
#
#     print("\nSemantic analysis completed successfully.")
#
#     # ===============================
#     # 4. PyTorch Code Generation
#     # ===============================
#     print("\n--- Generating PyTorch Code ---")
#
#     try:
#         generator = PyTorchCodeGenerator(listener)
#         #include_main = True if arguments.include_main else None
#
#         generated_code = generator.generate(
#             include_main=arguments.include_main
#         )
#
#         # Syntax validation for generated Python code
#         compile(generated_code, arguments.output, "exec")
#
#         output_dir = os.path.dirname(os.path.abspath(arguments.output))
#         if output_dir:
#             os.makedirs(output_dir, exist_ok=True)
#
#         with open(arguments.output, "w", encoding="utf8") as f:
#             f.write(generated_code)
#
#         print("PyTorch code generated successfully.")
#         print(f"Output file: {arguments.output}")
#
#         if arguments.print_code:
#             print("\n--- Generated PyTorch Code ---")
#             print(generated_code)
#
#     except ValueError as e:
#         print("\nCode generation failed.")
#         print(e)
#         return 1
#
#     except SyntaxError as e:
#         print("\nGenerated code has Python syntax error.")
#         print(type(e).__name__ + ":", e)
#
#         if arguments.debug:
#             traceback.print_exc()
#
#         return 1
#
#     except Exception as e:
#         print("\nUnexpected error during code generation.")
#         print(type(e).__name__ + ":", e)
#
#         if arguments.debug:
#             traceback.print_exc()
#
#         return 1
#
#     # ===============================
#     # 5. Optional AST Graph Display
#     # ===============================
#     if arguments.show_ast:
#         print("\n--- Showing AST Graph ---")
#
#         try:
#             if listener.ast.root is not None:
#                 show_ast(listener.ast.root)
#                 print("AST graph display completed.")
#             else:
#                 print("AST root is None. Tree was not built.")
#
#         except Exception as e:
#             print("\nAST graph display failed.")
#             print(type(e).__name__ + ":", e)
#
#             if arguments.debug:
#                 traceback.print_exc()
#
#     print("\nCompilation completed successfully.")
#     return 0
#
#
# if __name__ == "__main__":
#     argparser = argparse.ArgumentParser(
#         description="NNGraph DSL compiler: parse, semantic analysis, AST construction, and PyTorch code generation."
#     )
#
#     argparser.add_argument(
#         "-i",
#         "--input",
#         help="Input NNGraph source file",
#         default="input/test.txt"
#     )
#
#     argparser.add_argument(
#         "-o",
#         "--output",
#         help="Output generated PyTorch Python file",
#         default="generated_model.py"
#     )
#
#     argparser.add_argument(
#         "--include-main",
#         action="store_true",
#         help="Include a runnable if __name__ == '__main__' test block in generated PyTorch file"
#     )
#
#     argparser.add_argument(
#         "--print-code",
#         action="store_true",
#         help="Print generated PyTorch code in terminal"
#     )
#
#     argparser.add_argument(
#         "--show-ast",
#         action="store_true",
#         help="Display AST graph after successful semantic analysis and code generation"
#     )
#
#     argparser.add_argument(
#         "--debug",
#         action="store_true",
#         help="Print full traceback for unexpected errors"
#     )
#
#     args = argparser.parse_args()
#
#     raise SystemExit(main(args))