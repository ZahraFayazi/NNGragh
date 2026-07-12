def make_ast_subtree(ast, tree_node_ctx, node_value, keep_node=False):
    children = []
    raw_children_count = tree_node_ctx.getChildCount()
    valued_children_count = 0

    for i in range(raw_children_count):
        child = tree_node_ctx.getChild(i)
        if hasattr(child, "ast_value"):
            valued_children_count += 1

    ctx_is_not_compound = not hasattr(tree_node_ctx, "compound")
    dont_keep_node = (not keep_node) and ctx_is_not_compound

    if valued_children_count == 0 and raw_children_count == 0 and dont_keep_node:
        tree_node_ctx.ast_value = ast.make_node(tree_node_ctx.getText(), [])
        ast.root = tree_node_ctx.ast_value

    elif valued_children_count == 0 and raw_children_count == 1 and dont_keep_node:
        tree_node_ctx.ast_value = ast.make_node(tree_node_ctx.getChild(0).getText(), [])
        ast.root = tree_node_ctx.ast_value

    elif valued_children_count == 1 and dont_keep_node:
        for i in range(raw_children_count):
            child = tree_node_ctx.getChild(i)
            if hasattr(child, "ast_value"):
                tree_node_ctx.ast_value = child.ast_value
                ast.root = tree_node_ctx.ast_value
                break
    else:
        if not ctx_is_not_compound:
            children.append(ast.make_node("begin_scope_operator", []))

        for i in range(raw_children_count):
            child = tree_node_ctx.getChild(i)
            if hasattr(child, "ast_value"):
                children.append(child.ast_value)

        if (len(children) == 1 and not ctx_is_not_compound) or (len(children) == 0 and ctx_is_not_compound):
            if raw_children_count == 0:
                only_child = ast.make_node(tree_node_ctx.getText(), [])
                children.append(only_child)
            if raw_children_count == 1:
                only_child = ast.make_node(tree_node_ctx.getChild(0).getText(), [])
                children.append(only_child)

        if not ctx_is_not_compound:
            children.append(ast.make_node("end_scope_operator", []))

        sub_tree_pntr = ast.make_node(node_value, children)
        tree_node_ctx.ast_value = sub_tree_pntr
        ast.root = sub_tree_pntr