import numpy as np

"""
This is a simple implementation of the Interactive autodidactic school algortihm,
 recently proposed by M. Jahangiri et al. in the paper entitled
"Interactive autodidactic school: A new metaheuristic optimization algorithm for
 solving mathematical and structural design optimization problems" in Computers & Structures

Based on the paper, IAS is a population-based algorithm on the basis of
the interactions between students in an autodidactic school with the
goal of increasing their knowledge through a combination of
self-teaching/self-learning, interactive discussion, criticism
 and the competition. 

https://www.sciencedirect.com/science/article/abs/pii/S0045794920300717

"""

# Python Implementation by Amir Hosssein Namadchi
# Amir.Hossein.Namadchi@Gmail.com
# May 2020

class IAS:
    """
    Generates a Interactive Autodidactic School optimizer object

    Attributes:
        obj_func: Objective function to be optimized
        x_bounds: Lower and the upper bounds of design variables
        n_students: Number of the students
        n_iterations: Number of iterations
            
    """

    def __init__(self, obj_func, x_bounds: np.ndarray, n_students: int, n_iterations: int):

        self.obj_func = obj_func
        self.x_bounds = x_bounds
        self.n_students = n_students
        self.n_iterations = n_iterations
        self.school = self.__build_school()
        self.marks = self.__mark_school()
        self.x_size = len(x_bounds)
        self.history = {"f(x)":[], "iters":[]}

    def __repr__(self):
        return "IAS Object with {0} students and {1} iterations".format(self.n_students, self.n_iterations)


    def __generate_student(self):
        """
        Randomly generates a student based on the lower and upper bound values
        """
        return np.fromiter(
            map(lambda x: np.random.uniform(*x), self.x_bounds),
            dtype = np.float)


    def __build_school(self):
        """
        Builds a school based on the number of students
        """
        return np.array([self.__generate_student()
                         for n in range(self.n_students)])


    def __mark_school(self):
        """
        Marks the students based on their eligibility
        """
        return np.fromiter(map(self.obj_func, self.school), dtype = np.float)

    def do_optimization(self):
        """
        Performs Optimization process based on the interactions
        between students in an autodidactic school with the
        goal of increasing their knowledge through a combination
        of self-teaching/self-learning, interactive discussion,
         criticism, and the competition

        """

        p = 1
        while p <= self.n_iterations:

            q = 1
            while q <= self.n_students:
                # find the leader (L_S) here:
                L_S_id = np.argmin(self.marks) + 1
                L_S = self.school[L_S_id-1] 
                
                # 1 - Individual training session --------------------
                i = q
                # 1.1 - Don't want to include the leader as well as i
                j = np.random.choice(
                    np.setdiff1d(np.arange(self.n_students) + 1,
                                             [i, L_S_id]))
                # 1.2 - intrinsic competence (I_C)
                I_C = np.random.randint(1, high=3, size=2)
                T_S_i, T_S_j = self.school[i-1], self.school[j-1]
                
                T_s_S_i = T_S_i + np.random.rand(self.x_size)*(L_S - I_C[0]*T_S_i)
                T_s_S_j = T_S_j + np.random.rand(self.x_size)*(L_S - I_C[1]*T_S_j)
                
                f_s_i, f_s_j = self.obj_func(T_s_S_i), self.obj_func(T_s_S_j)
                
                if f_s_i < self.marks[i-1]:
                    self.school[i-1] = T_s_S_i
                    self.marks[i-1] = f_s_i
                    T_S_i = T_s_S_i
                
                if f_s_j < self.marks[j-1]:
                    self.school[j-1] = T_s_S_j
                    self.marks[j-1] = f_s_j
                    T_S_j = T_s_S_j
                

                # 2 - Collective training session --------------------                
                C_C = np.random.randint(1, high=3, size=2)
                # 2.2 - Collective capability of the group  (C_C_i_j)
                C_C_i_j = (1/(C_C[0]+C_C[1])*(C_C[0]*T_S_i + C_C[1]*T_S_j))
                
                T_s_S_i = T_S_i + np.random.rand(self.x_size)*(L_S - C_C[0]*C_C_i_j)
                T_s_S_j = T_S_j + np.random.rand(self.x_size)*(L_S - C_C[1]*C_C_i_j)
                f_s_i, f_s_j = self.obj_func(T_s_S_i), self.obj_func(T_s_S_j)
                
                if f_s_i < self.marks[i-1]:
                    self.school[i-1] = T_s_S_i
                    self.marks[i-1] = f_s_i
                    T_S_i = T_s_S_i
                
                if f_s_j < self.marks[j-1]:
                    self.school[j-1] = T_s_S_j
                    self.marks[j-1] = f_s_j
                    T_S_j = T_s_S_j        
                    
                # 3 - The challenge of the new student (N_S) -------------
                N_S = self.__generate_student()
                # 3.1 - modifier factors (M_F_1 & 2)
                M_F_1 = np.random.randint(0, 2, self.x_size)
                M_F_2 = 1 - M_F_1
                L_s_S = M_F_1*L_S + M_F_2*N_S
                f_L_s_S = self.obj_func(L_s_S)
                
                if f_L_s_S < self.marks[L_S_id-1]:
                    self.school[L_S_id-1] = L_s_S
                    self.marks[L_S_id-1] = f_L_s_S
                    L_S = L_s_S
                
                                            
                q += 1

            self.history['f(x)'].append(self.marks[L_S_id-1])
            self.history['iters'].append(p)
            p += 1

        print('Done!: f(x)=', self.marks[L_S_id-1], 'for x=', L_S)

        return (self.marks[L_S_id-1], L_S)        

