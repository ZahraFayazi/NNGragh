# Generated from C:/Users/Asus/Desktop/NNGragh/NNGraph.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .NNGraphParser import NNGraphParser
else:
    from NNGraphParser import NNGraphParser

# This class defines a complete listener for a parse tree produced by NNGraphParser.
class NNGraphListener(ParseTreeListener):

    # Enter a parse tree produced by NNGraphParser#program.
    def enterProgram(self, ctx:NNGraphParser.ProgramContext):
        pass

    # Exit a parse tree produced by NNGraphParser#program.
    def exitProgram(self, ctx:NNGraphParser.ProgramContext):
        pass


    # Enter a parse tree produced by NNGraphParser#modelDecl.
    def enterModelDecl(self, ctx:NNGraphParser.ModelDeclContext):
        pass

    # Exit a parse tree produced by NNGraphParser#modelDecl.
    def exitModelDecl(self, ctx:NNGraphParser.ModelDeclContext):
        pass


    # Enter a parse tree produced by NNGraphParser#inputDecl.
    def enterInputDecl(self, ctx:NNGraphParser.InputDeclContext):
        pass

    # Exit a parse tree produced by NNGraphParser#inputDecl.
    def exitInputDecl(self, ctx:NNGraphParser.InputDeclContext):
        pass


    # Enter a parse tree produced by NNGraphParser#outputDecl.
    def enterOutputDecl(self, ctx:NNGraphParser.OutputDeclContext):
        pass

    # Exit a parse tree produced by NNGraphParser#outputDecl.
    def exitOutputDecl(self, ctx:NNGraphParser.OutputDeclContext):
        pass


    # Enter a parse tree produced by NNGraphParser#graphBlock.
    def enterGraphBlock(self, ctx:NNGraphParser.GraphBlockContext):
        pass

    # Exit a parse tree produced by NNGraphParser#graphBlock.
    def exitGraphBlock(self, ctx:NNGraphParser.GraphBlockContext):
        pass


    # Enter a parse tree produced by NNGraphParser#graphStatement.
    def enterGraphStatement(self, ctx:NNGraphParser.GraphStatementContext):
        pass

    # Exit a parse tree produced by NNGraphParser#graphStatement.
    def exitGraphStatement(self, ctx:NNGraphParser.GraphStatementContext):
        pass


    # Enter a parse tree produced by NNGraphParser#nodeDecl.
    def enterNodeDecl(self, ctx:NNGraphParser.NodeDeclContext):
        pass

    # Exit a parse tree produced by NNGraphParser#nodeDecl.
    def exitNodeDecl(self, ctx:NNGraphParser.NodeDeclContext):
        pass


    # Enter a parse tree produced by NNGraphParser#layerType.
    def enterLayerType(self, ctx:NNGraphParser.LayerTypeContext):
        pass

    # Exit a parse tree produced by NNGraphParser#layerType.
    def exitLayerType(self, ctx:NNGraphParser.LayerTypeContext):
        pass


    # Enter a parse tree produced by NNGraphParser#edgeDecl.
    def enterEdgeDecl(self, ctx:NNGraphParser.EdgeDeclContext):
        pass

    # Exit a parse tree produced by NNGraphParser#edgeDecl.
    def exitEdgeDecl(self, ctx:NNGraphParser.EdgeDeclContext):
        pass


    # Enter a parse tree produced by NNGraphParser#label.
    def enterLabel(self, ctx:NNGraphParser.LabelContext):
        pass

    # Exit a parse tree produced by NNGraphParser#label.
    def exitLabel(self, ctx:NNGraphParser.LabelContext):
        pass


    # Enter a parse tree produced by NNGraphParser#paramList.
    def enterParamList(self, ctx:NNGraphParser.ParamListContext):
        pass

    # Exit a parse tree produced by NNGraphParser#paramList.
    def exitParamList(self, ctx:NNGraphParser.ParamListContext):
        pass


    # Enter a parse tree produced by NNGraphParser#param.
    def enterParam(self, ctx:NNGraphParser.ParamContext):
        pass

    # Exit a parse tree produced by NNGraphParser#param.
    def exitParam(self, ctx:NNGraphParser.ParamContext):
        pass


    # Enter a parse tree produced by NNGraphParser#value.
    def enterValue(self, ctx:NNGraphParser.ValueContext):
        pass

    # Exit a parse tree produced by NNGraphParser#value.
    def exitValue(self, ctx:NNGraphParser.ValueContext):
        pass


    # Enter a parse tree produced by NNGraphParser#shape.
    def enterShape(self, ctx:NNGraphParser.ShapeContext):
        pass

    # Exit a parse tree produced by NNGraphParser#shape.
    def exitShape(self, ctx:NNGraphParser.ShapeContext):
        pass


    # Enter a parse tree produced by NNGraphParser#configBlock.
    def enterConfigBlock(self, ctx:NNGraphParser.ConfigBlockContext):
        pass

    # Exit a parse tree produced by NNGraphParser#configBlock.
    def exitConfigBlock(self, ctx:NNGraphParser.ConfigBlockContext):
        pass


    # Enter a parse tree produced by NNGraphParser#configStatement.
    def enterConfigStatement(self, ctx:NNGraphParser.ConfigStatementContext):
        pass

    # Exit a parse tree produced by NNGraphParser#configStatement.
    def exitConfigStatement(self, ctx:NNGraphParser.ConfigStatementContext):
        pass



del NNGraphParser