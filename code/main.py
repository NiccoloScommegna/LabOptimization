import pycutest
prob = pycutest.import_problem('ROSENBR')  # esempio: Rosenbrock, se presente
# x0 = prob.x0
# f = lambda x: prob.obj(x)
# g = lambda x: prob.grad(x)
print(prob.x0)
print(prob.obj(prob.x0))
