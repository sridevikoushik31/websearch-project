from stat_parser import Parser
import random
import numpy as np
class Node:
    def __init__(self, left=None, right = None, label=None):
        self.left = left
        self.right = right
        self.label = label
    # pass

def ret_tree(sentence,rep,model):
    parser = Parser()

    tree_list = []
# sentence = "How are you"


    tree = parser.parse(sentence)
    #tree.draw()
    all_nodes = []


    def compute_tree_list(t, root_ptr1,rep,model):
        # if len(t.leaves()) == 2:
        # 	# tree_list.append(t.)
        # 	l = t.leaves()
        # 	# print l
        #print root_ptr1
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

#print "left id = %f" % left_id
 #           print "right id = %f" % right_id
            # print len(subts)
            left_tree = compute_tree_list(subts[0], left_id,rep,model)
            right_tree = compute_tree_list(subts[1], right_id,rep,model)
            if isinstance(left_tree, Node):
                left_id = left_tree.left
                rep[left_id]=np.transpose(model[left_id]).reshape([300,1])
   #             print rep[left_id].shape

            if isinstance(right_tree, Node):
                right_id = right_tree.left
                w=model.most_similar(positive=right_id,topn=1);
                rep[right_id]=np.transpose(model[right_id]).reshape([300,1])
  #              print rep[right_id].shape
            # print "root ptr.... = %f" % root_ptr
            tree_list.append({"ip1": left_id, "ip2": right_id, "op": root_ptr1})

            # return Node(left_tree, right_tree)

    compute_tree_list(tree, 10000,rep,model)
    print "Tree List",tree_list
    return tree_list,rep




