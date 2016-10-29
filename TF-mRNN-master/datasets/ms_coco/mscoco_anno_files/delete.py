import numpy
a = numpy.load("anno_list_mscoco_trainModelVal_m_RNN.npy")
for i in range(96,a.size):
	a = numpy.delete(a, 96)
print a
numpy.save('anno_list_mscoco_trainModelVal_m_RNN.npy',a)