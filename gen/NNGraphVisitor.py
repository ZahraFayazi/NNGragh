# Generated from C:/Users/Asus/Desktop/NNGragh/NNGraph.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .NNGraphParser import NNGraphParser
else:
    from NNGraphParser import NNGraphParser

# This class defines a complete generic visitor for a parse tree produced by NNGraphParser.

class NNGraphVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by NNGraphParser#program.
    def visitProgram(self, ctx:NNGraphParser.ProgramContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#modelDecl.
    def visitModelDecl(self, ctx:NNGraphParser.ModelDeclContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#inputDecl.
    def visitInputDecl(self, ctx:NNGraphParser.InputDeclContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#outputDecl.
    def visitOutputDecl(self, ctx:NNGraphParser.OutputDeclContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#graphBlock.
    def visitGraphBlock(self, ctx:NNGraphParser.GraphBlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#graphStatement.
    def visitGraphStatement(self, ctx:NNGraphParser.GraphStatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#nodeDecl.
    def visitNodeDecl(self, ctx:NNGraphParser.NodeDeclContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#layerType.
    def visitLayerType(self, ctx:NNGraphParser.LayerTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#edgeDecl.
    def visitEdgeDecl(self, ctx:NNGraphParser.EdgeDeclContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#label.
    def visitLabel(self, ctx:NNGraphParser.LabelContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#paramList.
    def visitParamList(self, ctx:NNGraphParser.ParamListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#param.
    def visitParam(self, ctx:NNGraphParser.ParamContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#value.
    def visitValue(self, ctx:NNGraphParser.ValueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#shape.
    def visitShape(self, ctx:NNGraphParser.ShapeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#configBlock.
    def visitConfigBlock(self, ctx:NNGraphParser.ConfigBlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#configStatement.
    def visitConfigStatement(self, ctx:NNGraphParser.ConfigStatementContext):
        return self.visitChildren(ctx)



del NNGraphParser