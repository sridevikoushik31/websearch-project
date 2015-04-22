from stat_parser import Parser
parser = Parser()

sentence = "The movie was great."

tree = parser.parse(sentence)

all_nodes = []

class Node:
	def __init__(self, left=None, right = None, leaf = False):
		self.left = left
		self.right = right
		self.leaf = leaf
	# pass
# tree_list = []

def compute_tree_list(t):
	if len(t.leaves()) == 2:
		# tree_list.append(t.)
		l = t.leaves()
		# print l
		# print l[0]
		# print l[1]
		return Node(l[0], l[1], True)
	elif len(t.leaves()) == 1:
		# tree_list.append(t.leaves())
		l = t.leaves()
		# print l[0]
		return Node(l[0], None, True)
	else:
		subts = list(t)
		# print len(subts)
		left_tree = compute_tree_list(subts[0])
		right_tree = compute_tree_list(subts[1])

		return Node(left_tree, right_tree)

n = compute_tree_list(tree)

def traverse(tree):
	# print "called.."
	if (tree.leaf == True):
		print tree.left, tree.right
	else:
		traverse(tree.left)
		traverse(tree.right)
# traverse(n)

traversal_stack = []

def traverse_reverse(tree):
	# print "called.."
	if (tree.leaf == True):
		traversal_stack.append(tree.left)
		traversal_stack.append(tree.right)

		# print tree.left, tree.right
	else:
		traversal_stack.append(tree.left)
		traversal_stack.append(tree.right)

		traverse(tree.left)
		traverse(tree.right)
def traverse_list(traverse_li):
	while len(traverse_li) > 0:
		popped = traverse_li.pop()
		if popped.leaf == True:
			print popped.left
			print popped.right

traverse_reverse(n)
traverse_list(traversal_stack)