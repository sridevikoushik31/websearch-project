from stat_parser import Parser
import random

parser = Parser()

sentence = "The movie was great"
# sentence = "How are you"


tree = parser.parse(sentence)
# tree.draw()
all_nodes = []

class Node:
	def __init__(self, left=None, right = None, label=None):
		self.left = left
		self.right = right
		self.label = label
	# pass
tree_list = []

def compute_tree_list(t, root_ptr1):
	# if len(t.leaves()) == 2:
	# 	# tree_list.append(t.)
	# 	l = t.leaves()
	# 	# print l
	print root_ptr1
	# 	# print l[0]
	# 	# print l[1]
	# 	return Node(l[0], l[1], True)
	if len(t.leaves()) == 1:
		# tree_list.append(t.leaves())
		l = t.leaves()
		# print l[0]
		return Node(l[0])
	else:
		subts = list(t)
		left_id = root_ptr1+1
		right_id = root_ptr1*2
		print "left id = %f" % left_id
		print "right id = %f" % right_id
		# print len(subts)
		left_tree = compute_tree_list(subts[0], left_id)
		right_tree = compute_tree_list(subts[1], right_id)
		if isinstance(left_tree, Node):
			left_id = left_tree.left
			
		if isinstance(right_tree, Node):
			right_id = right_tree.left
		# print "root ptr.... = %f" % root_ptr
		tree_list.append({"ip1": left_id, "ip2": right_id, "op": root_ptr1})

		# return Node(left_tree, right_tree)

x=random.random()*100
print "random... "
print x
compute_tree_list(tree, 10000)
import pdb
# pdb.set_trace()
print tree_list



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

# traverse_reverse(n)
# traverse_list(traversal_stack)