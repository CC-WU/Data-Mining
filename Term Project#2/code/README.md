if-than-else rule.ipynb can convert the prediction output from each model.

We do this by means of the Explanation Generator interpretability model.

First of all, we first find a good set of Subtrees from Random Forest, use some scoring mechanisms to find out the trees whose results are very close to the results of this model, and extract the rules of these trees, and we regard this rule as this model to Approximate Classification Rules.
