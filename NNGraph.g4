grammar NNGraph;


program : modelDecl graphBlock configBlock? EOF ;


modelDecl : 'model' ID '{' inputDecl outputDecl '}' ;

inputDecl : 'input' ID ':' 'tensor' '(' shape ')' ;

outputDecl : 'output' ID ;


graphBlock : 'graph' '{' graphStatement* '}' ;

graphStatement : nodeDecl | edgeDecl ;


nodeDecl : 'node' ID ':' layerType '(' paramList? ')' ;

layerType : ID ;


edgeDecl : 'edge' ID '->' ID label? ;

label : '[' 'label' '=' STRING ']' ;


paramList : param (',' param)* ;

param : ID '=' value ;

value: INT | FLOAT | STRING | BOOL | 'None' | '(' shape ')' ;


shape : INT (',' INT)* ;


configBlock : 'config' '{' configStatement* '}' ;

configStatement : ID '=' value ;


BOOL  : 'true' | 'false';
ID    : [a-zA-Z_][a-zA-Z0-9_]*;
FLOAT : [0-9]+ '.' [0-9]+;
INT   : [0-9]+;
STRING: '"' .*? '"';

COMMENT_LINE : '//' ~[\r\n]* -> skip;
COMMENT_BLOCK: '/*' .*? '*/' -> skip;
WS    : [ \t\r\n]+ -> skip;