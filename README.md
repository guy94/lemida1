# ML1
## Please track master branch.

## Abstract
In the decision tree algorithm, samples are routed through the decision tree according to the results of the conditions at each of the splitting nodes. A prediction is then provided for each sample based on the leaf that the sample is routed to.
A potential flaw with the decision tree algorithm is that the routing through the tree is binary. Hence, at each stage, one side of the tree is rendered irrelevant. The problem with this is that this method doesn’t account for the inherent uncertainty of the decision tree splits. Examples of situations that should generate uncertainty: 
* A sample is close to the decision boundary
* The split condition is based on a few training observations.


## Task
Modify dklearn decision tree classiffier to employ "soft splits". Each sample can be routed to either side of each split with a certain probability - alpha=0.1 side A, (1-alpha)=0.9 side B.

## Proposed Solution
Train model without any modification. Before inference, for each sample, extract the fitted tree split nodes features and thresholds values. 
For each feature of the sample, manipulate the corresponding tree nodes and change the threshold value of that node so when the sample arrives to that node during inference it will be redirected to the oposite direction of the original one (with certain probability). For example:

Given a tree node with the slected feature to split by - 'age' and all the samples that have age value smaller than 10 are redirected left. 
Tested sample has 'age'=9 which means it will be directed left. With probability=0.1, change the value of the node (value=10) to be smaller than 9 so that same sample will be redirected right.
