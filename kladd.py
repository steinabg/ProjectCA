import CAenvironment as CAenv

parameters = CAenv.import_parameters()

nj = parameters['nj']
print(nj)
print(str(nj))

for particle_type in range(nj):
    s = 'q_cj[y,x,' + str(particle_type) + ']'
    print(s)
    print(parameters[s])
