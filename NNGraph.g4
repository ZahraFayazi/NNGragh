grammar NNGraph;

// ========== ROOT ==========
program : modelDecl graphBlock configBlock? EOF ;

// ========== MODEL ==========
modelDecl : 'model' ID '{' inputDecl outputDecl '}' ;

inputDecl : 'input' ID ':' 'tensor' '(' shape ')' ;

outputDecl : 'output' ID ;

// ========== GRAPH ==========
graphBlock : 'graph' '{' graphStatement* '}' ;

graphStatement : nodeDecl | edgeDecl ;

// ========== NODE ==========
nodeDecl : 'node' ID ':' layerType '(' paramList? ')' ;

layerType : ID ;

// ========== EDGE ==========
edgeDecl : 'edge' ID '->' ID label? ;

label : '[' 'label' '=' STRING ']' ;

// ========== PARAMS ==========
paramList : param (',' param)* ;

param : ID '=' value ;

value: INT | FLOAT | STRING | BOOL | 'None' | '(' shape ')' ;

// ========== SHAPE ==========
shape : INT (',' INT)* ;

// ========== CONFIG ==========
configBlock : 'config' '{' configStatement* '}' ;

configStatement : ID '=' value ;

// ========== LEXER ==========
BOOL  : 'true' | 'false';
ID    : [a-zA-Z_][a-zA-Z0-9_]*;
FLOAT : [0-9]+ '.' [0-9]+;
INT   : [0-9]+;
STRING: '"' .*? '"';

COMMENT_LINE : '//' ~[\r\n]* -> skip;
COMMENT_BLOCK: '/*' .*? '*/' -> skip;
WS    : [ \t\r\n]+ -> skip;