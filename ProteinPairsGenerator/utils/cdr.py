from constraint import *

def getLightCDR(seq : str):
    problem = Problem()
    problem.addVariable("L1_start", range(20, 28))
    problem.addVariable("L1_size", range(10, 17 + 1))
    problem.addVariable("L3_size", range(7, 12 + 1))

    # Add constraints for L1
    L1_start_seq = "C"
    L1_end_seq = ["WYQ", "WLQ", "WFQ", "WYL"]
    problem.addConstraint(lambda x, y: seq[x - 1] == L1_start_seq, 
                          ("L1_start", "L1_size"))
    problem.addConstraint(lambda x, y: seq[x + y: x + y + 3] in L1_end_seq,
                          ("L1_start", "L1_size"))

    # Add constraints for L2
    L1_L2_diff = 15 # 16 - 1
    L2_size = 7
    L2_start_seq = ["IY", "VY", "IK", "IF"]
    problem.addConstraint(lambda x, y: 
                          seq[x + y + L1_L2_diff - 2 : x + y + L1_L2_diff] \
                          in L2_start_seq, 
                          ("L1_start", "L1_size"))

    # Add constraints for L3
    L2_L3_diff = 32 # 33 - 1
    L3_start_seq = "C"
    L3_end_seq_pre = "FG"
    L3_end_seq_post = "G"
    problem.addConstraint(lambda x, y: seq[x + y + L1_L2_diff + L2_size + L2_L3_diff - 1] == L3_start_seq, 
                          ("L1_start", "L1_size"))
    problem.addConstraint(lambda x, y, z: \
                          seq[x + y + L1_L2_diff + L2_size + L2_L3_diff + z - 2 : \
                              x + y + L1_L2_diff + L2_size + L2_L3_diff + z] \
                          == L3_end_seq_pre, 
                          ("L1_start", "L1_size", "L3_size"))
    problem.addConstraint(lambda x, y, z: \
                          seq[x + y + L1_L2_diff + L2_size + L2_L3_diff + z + 1] \
                          == L3_end_seq_post, 
                          ("L1_start", "L1_size", "L3_size"))

    solutions = problem.getSolutions()

    # Assert uniqueness
    assert len(solutions) == 1

    unique_solution = solutions[0]

    L1_start = unique_solution["L1_start"]
    L1_end = L1_start + unique_solution["L1_size"]
    L2_start = L1_end + L1_L2_diff
    L2_end = L2_start + L2_size
    L3_start = L2_end + L2_L3_diff
    L3_end = L3_start + unique_solution["L3_size"]
    
    yield slice(L1_start, L1_end)
    yield slice(L2_start, L2_end) 
    yield slice(L3_start, L3_end)


def getHeavyCDR(seq : str):
    problem = Problem()
    problem.addVariable("H1_start", range(22, 30))
    problem.addVariable("H1_size", range(10, 12 + 1))
    problem.addVariable("H2_size", range(16, 19 + 1))
    problem.addVariable("H3_size", range(2, 25 + 1))

    # Add constraints for H1
    H1_start_seq = "C"
    H1_end_seq = ["WV", "WI", "WA"]
    problem.addConstraint(lambda x, y: seq[x - 4] == H1_start_seq, 
                          ("H1_start", "H1_size"))
    problem.addConstraint(lambda x, y: seq[x + y: x + y + 2] in H1_end_seq,
                          ("H1_start", "H1_size"))


    # Add constraints for H2
    H1_H2_diff = 14 # 15 - 1
    H2_end_seq1 = ["K", "I", "V", "F", "T", "S", "A"]
    H2_end_seq2 = ["RL", "AT"]
    problem.addConstraint(lambda x, y, z: 
                          seq[x + y + H1_H2_diff + z + 1] in H2_end_seq1 or \
                          seq[x + y + H1_H2_diff + z + 1 : \
                              x + y + H1_H2_diff + z + 3] in H2_end_seq2, 
                          ("H1_start", "H1_size", "H2_size"))

    # Add constraints for H3
    H2_H3_diff = 32 # 15 - 1
    H3_start_seq = "C"
    H3_end_seq_pre = "WG"
    H3_end_seq_post = "G"
    problem.addConstraint(lambda x, y, z: 
                          seq[x + y + H1_H2_diff + z + H2_H3_diff - 3] \
                          ==  H3_start_seq,
                          ("H1_start", "H1_size", "H2_size"))
    problem.addConstraint(lambda x, y, z, a: 
                          seq[x + y + H1_H2_diff + z + H2_H3_diff + a - 2 : \
                              x + y + H1_H2_diff + z + H2_H3_diff + a] \
                          ==  H3_end_seq_pre,
                          ("H1_start", "H1_size", "H2_size", "H3_size"))
    problem.addConstraint(lambda x, y, z, a: 
                          seq[x + y + H1_H2_diff + z + H2_H3_diff + a + 1] \
                          ==  H3_end_seq_post,
                          ("H1_start", "H1_size", "H2_size", "H3_size"))

    solutions = problem.getSolutions()

    # Assert uniqueness
    assert len(solutions) == 1

    unique_solution = solutions[0]

    H1_start = unique_solution["H1_start"]
    H1_end = H1_start + unique_solution["H1_size"]
    H2_start = H1_end + H1_H2_diff
    H2_end = H2_start + unique_solution["H3_size"]
    H3_start = H2_end + H2_H3_diff
    H3_end = H3_start + unique_solution["L3_size"]
    
    yield slice(H1_start, H1_end) 
    yield slice(H2_start, H2_end) 
    yield slice(H3_start, H3_end)
