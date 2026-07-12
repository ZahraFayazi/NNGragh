class AST:
    def __init__(self):
        self.root = None
        self.current_number = 0

    class TreeNode:
        def __init__(self, value, children, number):
            self.value = value
            self.children = children
            self.number = number
            self.attributes_dictionary = dict()

        def add_attribute(self, key, value):
            self.attributes_dictionary[key] = value

    def make_node(self, value, children):
        tree_node = self.TreeNode(value, children, self.current_number)
        self.current_number += 1
        return tree_node

    def traverse_ast(self, root_node):
        traversal = []
        for child in root_node.children:
            traversal.extend(self.traverse_ast(child))
        traversal.append(root_node.value)
        return traversal