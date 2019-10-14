#coding=utf8


##目的：提取出CDK的无‘X’的peptide，使用model/protvec的数据进行svm/mlp训练预测


from gensim.models import word2vec
import numpy as np
from sklearn import svm
import random
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pyplot as plt

def extract(*kw):
	print kw
	filename1,filename2,mode,model,algorithm=kw[0]
	with open(filename1) as in_f:
		with open(filename2) as in_f2:
			#######################去除SSright蛋白质结构预测结果#################
			no_line3 = []
			for m, n in enumerate(in_f.readlines()):
				if (m - 3) % 6 != 0:
					no_line3.append(n.strip('\n'))
				# print len(no_line3)


				#################检查是否有蛋白质长度小于13##############
			b = 0
			for i in no_line3:
				if '>' not in i and len(i) <= 13:
					b += 1
				# print 0

				#######################将各种预测信息分到不同字典中################
			all_motif = [];
			c = 0;d = 0;e = 0;
			f = 0;g = 0;h = 0
			positive1 = {};positive2 = {};positive3 = {}
			all_dataset1 = {};all_dataset2 = {};all_dataset3 = {}
			left = 6;right = 7															#########测试对不同激酶取不同长度peptide
			for i in in_f2.readlines():
				if 'PKC' in i.split('\t')[1]:
					h += 1
				for m, n in enumerate(no_line3):
					if '|' in n:
						if 'PKC' in i.split('\t')[1] and i.split('\t')[6] == n.split('|')[
							1]:  #############原代码判断条件错误，没有加上CDK激酶这一限制条件
							g += 1
							for pos, aa in enumerate(no_line3[m + 1]):
								if aa in ['Y', 'S', 'T']:
									if pos > 5:
										if len(no_line3[m + 1]) - pos > left:
											if len(no_line3[m + 1]) - pos == right:
												all_dataset1[no_line3[m + 1][pos - left:]] = no_line3[m + 2][pos - left:]
												all_dataset2[no_line3[m + 1][pos - left:]] = no_line3[m + 3][pos - left:]
												all_dataset3[no_line3[m + 1][pos - left:]] = no_line3[m + 4][pos - left:]

											else:
												all_dataset1[no_line3[m + 1][pos - left:pos + right]] = no_line3[m + 2][
																										pos - left:pos + right]
												all_dataset2[no_line3[m + 1][pos - left:pos + right]] = no_line3[m + 3][
																										pos - left:pos + right]
												all_dataset3[no_line3[m + 1][pos - left:pos + right]] = no_line3[m + 4][
																										pos - left:pos + right]
										else:
											all_dataset1[
												no_line3[m + 1][pos - left:] + (pos + right - len(no_line3[m + 1])) * 'X'] = \
											no_line3[m + 2][pos - left:] + (pos + right - len(no_line3[m + 1])) * 'X'
											all_dataset2[
												no_line3[m + 1][pos - left:] + (pos + right - len(no_line3[m + 1])) * 'X'] = \
											no_line3[m + 3][pos - left:] + (pos + right - len(no_line3[m + 1])) * 'X'
											all_dataset3[
												no_line3[m + 1][pos - left:] + (pos + right - len(no_line3[m + 1])) * 'X'] = \
											no_line3[m + 4][pos - left:] + (pos + right - len(no_line3[m + 1])) * 'X'
									else:
										all_dataset1['X' * (left - pos) + no_line3[m + 1][:pos + right]] = 'X' * (
										left - pos) + no_line3[m + 2][:pos + right]
										all_dataset2['X' * (left - pos) + no_line3[m + 1][:pos + right]] = 'X' * (
										left - pos) + no_line3[m + 3][:pos + right]
										all_dataset3['X' * (left - pos) + no_line3[m + 1][:pos + right]] = 'X' * (
										left - pos) + no_line3[m + 4][:pos + right]

							postion = int(i.split('\t')[9][1:])
							if postion > left:
								if len(no_line3[m + 1]) - postion > 5:
									if len(no_line3[m + 1]) - postion == left:
										c += 1  # 找出motif刚好在蛋白质末端的数量
										positive1[no_line3[m + 1][postion - right:]] = no_line3[m + 2][postion - right:]
										positive2[no_line3[m + 1][postion - right:]] = no_line3[m + 3][postion - right:]
										positive3[no_line3[m + 1][postion - right:]] = no_line3[m + 4][postion - right:]
										# if len(n[int(i.split('\t')[9][1:]-left:])!=13:
										# print n[int(i.split('\t')[9][1:]-left:]
										if no_line3[m + 1][postion - right:] not in all_motif:  # 添加无重复的motif
											all_motif.append(no_line3[m + 1][postion - right:])
									else:
										d += 1  # 找出motif在蛋白质中间的数量
										positive1[no_line3[m + 1][postion - right:postion + left]] = no_line3[m + 2][
																									 postion - right:postion + left]
										positive2[no_line3[m + 1][postion - right:postion + left]] = no_line3[m + 3][
																									 postion - right:postion + left]
										positive3[no_line3[m + 1][postion - right:postion + left]] = no_line3[m + 4][
																									 postion - right:postion + left]
										# if len(n[int(i.split('\t')[9][1:]-left:int(i.split('\t')[9][1:]+6])!=13:
										# print n[int(i.split('\t')[9][1:]-left:int(i.split('\t')[9][1:]+6]
										if no_line3[m + 1][postion - right:postion + left] not in all_motif:  # 添加无重复的motif
											all_motif.append(no_line3[m + 1][postion - right:postion + left])
								else:
									e += 1  # 找出在蛋白质末端且需要添加X的motif的数量
									positive1[
										no_line3[m + 1][postion - right:] + (postion + left - len(no_line3[m + 1])) * 'X'] = \
									no_line3[m + 2][postion - right:] + (postion + left - len(
										no_line3[m + 1])) * 'X'  # 为什么+9才能正常运行？
									positive2[
										no_line3[m + 1][postion - right:] + (postion + left - len(no_line3[m + 1])) * 'X'] = \
									no_line3[m + 3][postion - right:] + (postion + left - len(no_line3[m + 1])) * 'X'
									positive3[
										no_line3[m + 1][postion - right:] + (postion + left - len(no_line3[m + 1])) * 'X'] = \
									no_line3[m + 4][postion - right:] + (postion + left - len(no_line3[m + 1])) * 'X'
									if len(no_line3[m + 1][postion - right:] + (postion + left - len(no_line3[m + 1])) * 'X') != 13:
										print no_line3[m + 1][postion - right:] + (postion + left - len(no_line3[m + 1])) * 'X'  # 输出异常数据

									if no_line3[m + 1][postion - right:] + (
											postion + left - len(no_line3[m + 1])) * 'X' not in all_motif:  # 添加无重复的motif
										all_motif.append(no_line3[m + 1][postion - right:] + (postion + left - len(no_line3[m + 1])) * 'X')
							else:
								f += 1  # 找出在蛋白质前端（添加X）的motif的数量

								positive1['X' * (right - postion) + no_line3[m + 1][:postion + left]] = 'X' * (right - postion) + no_line3[m + 2][:postion + left]
								positive2['X' * (right - postion) + no_line3[m + 1][:postion + left]] = 'X' * (right - postion) + no_line3[m + 3][:postion + left]
								positive3['X' * (right - postion) + no_line3[m + 1][:postion + left]] = 'X' * (right - postion) + no_line3[m + 4][:postion + left]
								# if len('X'*(left-postion+n[:int(i.split('\t')[9][1:]+6])!=13:
								# print 'X'*(left-postion+n[:int(i.split('\t')[9][1:]+6]
								if 'X' * (right - postion) + no_line3[m + 1][:postion + left] not in all_motif:
									all_motif.append('X' * (right - postion) + no_line3[m + 1][:postion + left])

			print g, h  ########源文件中某个激酶对应的底物的所有条目

			in_f2.close()
		in_f.close()
	#################鉴定是否存在异常数据########################
	error = 0
	for i in positive1.keys():
		if len(i) != 13:
			error += 1
			# print i,all_dataset1[i]
			del positive1[i]
	for i in positive2.keys():
		if len(i) != 13:
			del positive2[i]
	for i in positive3.keys():
		if len(i) != 13:
			del positive3[i]
	print 'the number of abnormal data:', error
	print 'all identified S/Y/T sites:', len(positive1.keys());
	print 'all (un)identified S/Y/T sites:', len(all_dataset3.values())  # 结果有1left303个motif，但在原始文件中存在1leftleft33个位点，因此应该有430个重复位点
	print len(all_motif)  ##############所有无重复的磷酸化motif

	final_vec1 = []
	T_positive = [acid for acid in positive1.keys() if acid[left] in ['T', 'Y', 'S']]
	T_positive2=[i for i in T_positive if 'X' not in i]
	final_vec2 = []
	T_all = [acid for acid in all_dataset1.keys() if acid[left] in ['T', 'Y', 'S']]
	T_all_negative = [i for i in T_all if i not in T_positive]
	T_all_negative2=[i for i in T_all_negative if 'X' not in i]
	T_negative2 = random.sample(T_all_negative2, len(T_positive2))
	all_dataset = T_negative2 + T_positive2
	print len(T_negative2)
	
	########################H为helix的aa，E为beta的aa，C为coil的aa；B为深埋内部的aa，M为位于中部aa，E为暴露外界的aa；*为无序aa，.为有序aa#####################################
	structure = ['C', 'H', 'E', 'X'];solvent = ['X', 'E', 'M', 'B'];order = ['X', '*', '.']
	
	###############创建20个分字典及一个总字典######################
	C = {'C': 9, 'S': -1, 'T': -1, 'P': -3, 'A': 0, 'G': -3, 'N': -3, 'D': -3, 'E': -4, 'Q': -3, 'H': -3, 'R': -3,'K': -3, 'M': -1, 'I': -1, 'L': -1, 'V': -1, 'F': -2, 'Y': -2, 'W': -2}
	S = {'C': -1, 'S': 4, 'T': 1, 'P': -1, 'A': 1, 'G': 0, 'N': 1, 'D': 0, 'E': 0, 'Q': 0, 'H': -1, 'R': -1, 'K': 0,'M': -1, 'I': -2, 'L': -2, 'V': -2, 'F': -2, 'Y': -2, 'W': -3}
	T = {'C': -1, 'S': 1, 'T': 4, 'P': 1, 'A': -1, 'G': 1, 'N': 0, 'D': 1, 'E': 0, 'Q': 0, 'H': 0, 'R': -1, 'K': 0,'M': -1, 'I': -2, 'L': -2, 'V': -2, 'F': -2, 'Y': -2, 'W': -3}
	P = {'C': -3, 'S': -1, 'T': 1, 'P': 7, 'A': -1, 'G': -2, 'N': -1, 'D': -1, 'E': -1, 'Q': -1, 'H': -2, 'R': -2, 'K': -1, 'M': -2, 'I': -3, 'L': -3, 'V': -2, 'F': -4, 'Y': -3, 'W': -4}
	A = {'C': 0, 'S': 1, 'T': -1, 'P': -1, 'A': 4, 'G': 0, 'N': -1, 'D': -2, 'E': -1, 'Q': -1, 'H': -2, 'R': -1,'K': -1, 'M': -1, 'I': -1, 'L': -1, 'V': -2, 'F': -2, 'Y': -2, 'W': -3}
	G = {'C': -3, 'S': 0, 'T': 1, 'P': -2, 'A': 0, 'G': 6, 'N': -2, 'D': -1, 'E': -2, 'Q': -2, 'H': -2, 'R': -2,'K': -2, 'M': -3, 'I': -4, 'L': -4, 'V': 0, 'F': -3, 'Y': -3, 'W': -2}
	N = {'C': -3, 'S': 1, 'T': 0, 'P': -2, 'A': -2, 'G': 0, 'N': 6, 'D': 1, 'E': 0, 'Q': 0, 'H': -1, 'R': 0, 'K': 0, 'M': -2, 'I': -3, 'L': -3, 'V': -3, 'F': -3, 'Y': -2, 'W': -4}
	D = {'C': -3, 'S': 0, 'T': 1, 'P': -1, 'A': -2, 'G': -1, 'N': 1, 'D': 6, 'E': 2, 'Q': 0, 'H': -1, 'R': -2, 'K': -1, 'M': -3, 'I': -3, 'L': -4, 'V': -3, 'F': -3, 'Y': -3, 'W': -4}
	E = {'C': -4, 'S': 0, 'T': 0, 'P': -1, 'A': -1, 'G': -2, 'N': 0, 'D': 2, 'E': 5, 'Q': 2, 'H': 0, 'R': 0, 'K': 1, 'M': -2, 'I': -3, 'L': -3, 'V': -3, 'F': -3, 'Y': -2, 'W': -3}
	Q = {'C': -3, 'S': 0, 'T': 0, 'P': -1, 'A': -1, 'G': -2, 'N': 0, 'D': 0, 'E': 2, 'Q': 5, 'H': 0, 'R': 1, 'K': 1,'M': 0, 'I': -3, 'L': -2, 'V': -2, 'F': -3, 'Y': -1, 'W': -2}
	H = {'C': -3, 'S': -1, 'T': 0, 'P': -2, 'A': -2, 'G': -2, 'N': 1, 'D': 1, 'E': 0, 'Q': 0, 'H': 8, 'R': 0, 'K': -1, 'M': -2, 'I': -3, 'L': -3, 'V': -2, 'F': -1, 'Y': 2, 'W': -2}
	R = {'C': -3, 'S': -1, 'T': -1, 'P': -2, 'A': -1, 'G': -2, 'N': 0, 'D': -2, 'E': 0, 'Q': 1, 'H': 0, 'R': 5, 'K': 2, 'M': -1, 'I': -3, 'L': -2, 'V': -3, 'F': -3, 'Y': -2, 'W': -3}
	K = {'C': -3, 'S': 0, 'T': 0, 'P': -1, 'A': -1, 'G': -2, 'N': 0, 'D': -1, 'E': 1, 'Q': 1, 'H': -1, 'R': 2, 'K': 5, 'M': -1, 'I': -3, 'L': -2, 'V': -3, 'F': -3, 'Y': -2, 'W': -3}
	M = {'C': -1, 'S': -1, 'T': -1, 'P': -2, 'A': -1, 'G': -3, 'N': -2, 'D': -3, 'E': -2, 'Q': 0, 'H': -2, 'R': -1, 'K': -1, 'M': 5, 'I': 1, 'L': 2, 'V': -2, 'F': 0, 'Y': -1, 'W': -1}
	I = {'C': -1, 'S': -2, 'T': -2, 'P': -3, 'A': -1, 'G': -4, 'N': -3, 'D': -3, 'E': -3, 'Q': -3, 'H': -3, 'R': -3,'K': -3, 'M': 1, 'I': 4, 'L': 2, 'V': 1, 'F': 0, 'Y': -1, 'W': -3}
	L = {'C': -1, 'S': -2, 'T': -2, 'P': -3, 'A': -1, 'G': -4, 'N': -3, 'D': -4, 'E': -3, 'Q': -2, 'H': -3, 'R': -2,'K': -2, 'M': 2, 'I': 2, 'L': 4, 'V': 3, 'F': 0, 'Y': -1, 'W': -2}
	V = {'C': -1, 'S': -2, 'T': -2, 'P': -2, 'A': 0, 'G': -3, 'N': -3, 'D': -3, 'E': -2, 'Q': -2, 'H': -3, 'R': -3,'K': -2, 'M': 1, 'I': 3, 'L': 1, 'V': 4, 'F': -1, 'Y': -1, 'W': -3}
	F = {'C': -2, 'S': -2, 'T': -2, 'P': -4, 'A': -2, 'G': -3, 'N': -3, 'D': -3, 'E': -3, 'Q': -3, 'H': -1, 'R': -3,'K': -3, 'M': 0, 'I': 0, 'L': 0, 'V': -1, 'F': 6, 'Y': 3, 'W': 1}
	Y = {'C': -2, 'S': -2, 'T': -2, 'P': -3, 'A': -2, 'G': -3, 'N': -2, 'D': -3, 'E': -2, 'Q': -1, 'H': 2, 'R': -2,'K': -2, 'M': -1, 'I': -1, 'L': -1, 'V': -1, 'F': 3, 'Y': 7, 'W': 2}
	W = {'C': -2, 'S': -3, 'T': -3, 'P': -4, 'A': -3, 'G': -2, 'N': -4, 'D': -4, 'E': -3, 'Q': -2, 'H': -2, 'R': -3,'K': -3, 'M': -1, 'I': -3, 'L': -2, 'V': -3, 'F': 1, 'Y': 2, 'W': 11}
	
	blosum62 = {'C': C, 'S': S, 'T': T, 'P': P, 'A': A, 'G': G, 'N': N, 'D': D, 'E': E, 'Q': Q, 'H': H, 'R': R, 'K': K,'M': M, 'I': I, 'L': L, 'V': V, 'F': F, 'Y': Y, 'W': W}
	
	######################将X取其他aa的均值加入到各个分字典中，另创建一个分字典并在总字典中添加X分字典################
	X = {};maxi = 0
	for x in blosum62.keys():
		he = 0
		for m in blosum62[x].values():
			he += m
		blosum62[x]['X'] = he * 1.0 / 20
		X[x] = he / 20.0
		maxi += X[x]
	X['X'] = maxi * 1.0 / 20
	blosum62['X'] = X
	print blosum62.values();
	print X
	
	#####################找出前10个neighbors，并求其中阳性数据的比例############################
	KNN = {}
	for x in all_dataset:
		# x='TMVKQMTDVLLTP'
		distance = {};k_neighbors = []
		for m in all_dataset:
			if m != x:
				dist = 0
				for i in range(13):
					dist += blosum62[x[i]][m[i]]
				distance[m] = dist
		# print distance
		k = 10
		sorted_distance = sorted(zip(distance.itervalues(),distance.iterkeys()))  # zip函数作用于可迭代对象，将对应的元素打包成一个个元组，然后返回有这些元素组成的列表，sort函数按照列表中元组的第一个元素的大小从小到大排序
		k_neighbors = sorted_distance[-k:]
		# print k_neighbors
		
		number = 0
		for i in k_neighbors:
			if i[1] in T_positive2:
				number += 1
		KNN[x] = number * 1.0 / k



	if mode=='no_overlap':
		model = word2vec.Word2Vec.load(model)
		aa = ['C', 'S', 'T', 'P', 'A', 'G', 'N', 'D', 'E', 'Q', 'H', 'R', 'K', 'M', 'I', 'L', 'V', 'F', 'Y', 'W']
		words = [];features = {}
		for i in aa:
			for x in aa:
				for y in aa:
					words.append(i+x+y)
		print len(words)

		for i in words:
			features[i]=model[i]
		print len(features)

		for pep in T_positive2:
			i = 0;gram = [];a = np.zeros((1, 100), dtype=float);vec=[]
			while i < len(pep):
				if i <= len(pep) - 3:
					gram.append(pep[i:i + 3])
					i += 3
				else:
					gram.append(pep[i:])
					i = len(pep)
			# print gram
			for x in gram:
				for i, num in features.items():
					if i == x:
						a += num
			vec=a.tolist()[0]
			##KNN数据
			vec.append(KNN[pep])
			####基于二级结构
			for i in positive1[pep]:
				for x in structure:
					if i==x:
						if x!='X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			####基于疏水性
			for i in positive2[pep]:
				for x in solvent:
					if i==x:
						if x!='X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			####基于有序性
			for i in positive3[pep]:
				for x in order:
					if i==x:
						if x!='X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			final_vec1.append(vec)

		# peptide=['ASTGVDYTSAMLWT','ASGTGVHMTYSLWE']
		for pep in T_negative2:
			i = 0;gram = [];a = np.zeros((1, 100), dtype=float);vec=[]
			while i < len(pep):
				if i <= len(pep) - 3:
					gram.append(pep[i:i + 3])
					i += 3
				else:
					gram.append(pep[i:])
					i = len(pep)
			# print gram
			for x in gram:
				for i, num in features.items():
					if i == x:
						a += num
			vec=a.tolist()[0]
			##KNN数据
			vec.append(KNN[pep])
			####基于二级结构
			for i in all_dataset1[pep]:
				for x in structure:
					if i==x:
						if x!='X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			####基于疏水性
			for i in all_dataset2[pep]:
				for x in solvent:
					if i==x:
						if x!='X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			####基于有序性
			for i in all_dataset3[pep]:
				for x in order:
					if i==x:
						if x!='X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			final_vec2.append(vec)


	elif mode=='overlap':
		with open(r'E:\python_project\original_data\protVec_100d_3grams.csv')as f:
			header=f.readline();lst=f.readlines()
		f.close()
		
		
		for pep in T_positive2:
			gram = [];a = np.zeros((1, 100), dtype=float);vec=[]
			for i in range(len(pep) - 3):
				gram.append(pep[i:i + 3])
			gram.append(pep[-3:])
			#print gram
			for x in gram:
				for i in lst:
					if i.strip('\n').split('\t')[0]==x:
						a+=np.array(i.strip('\n').split('\t')[1:],dtype=float)
			vec=a.tolist()[0]
			##KNN数据
			vec.append(KNN[pep])
			####基于二级结构
			for i in positive1[pep]:
				for x in structure:
					if i==x:
						if x!='X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			####基于疏水性
			for i in positive2[pep]:
				for x in solvent:
					if i==x:
						if x!='X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			####基于有序性
			for i in positive3[pep]:
				for x in order:
					if i==x:
						if x!='X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			final_vec1.append(vec)
			



		#peptide=['ASTGVDYTSAMLWT','ASGTGVHMTYSLWE']
		for pep in T_negative2:
			gram=[];a = np.zeros((1, 100), dtype=float);vec=[]
			for i in range(len(pep)-3):
				gram.append(pep[i:i+3])
			gram.append(pep[-3:])
			#print gram
			for x in gram:
				for i in lst:
					if i.strip('\n').split('\t')[0]==x:
						a+=np.array(i.strip('\n').split('\t')[1:],dtype=float)
			vec=a.tolist()[0]
			##KNN数据
			vec.append(KNN[pep])
			####基于二级结构
			for i in all_dataset1[pep]:
				for x in structure:
					if i==x:
						if x!='X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			####基于疏水性
			for i in all_dataset2[pep]:
				for x in solvent:
					if i==x:
						if x!='X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			####基于有序性
			for i in all_dataset3[pep]:
				for x in order:
					if i==x:
						if x!='X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			final_vec2.append(vec)
	
	
	elif mode == 'our_overlap':
		model = word2vec.Word2Vec.load(model)
		aa = ['C', 'S', 'T', 'P', 'A', 'G', 'N', 'D', 'E', 'Q', 'H', 'R', 'K', 'M', 'I', 'L', 'V', 'F', 'Y', 'W']
		words = [];features = {}
		for i in aa:
			for x in aa:
				for y in aa:
					words.append(i + x + y)
		print len(words)
		for i in words:
			features[i] = model[i]
		
		for pep in T_positive2:
			gram = [];
			a = np.zeros((1, 100), dtype=float)
			for i in range(len(pep) - 3):
				gram.append(pep[i:i + 3])
			gram.append(pep[-3:])
		
			for x in gram:
				for i, num in features.items():
					if i == x:
						a += num
			final_vec1.append(a.tolist()[0])
		
		for pep in T_negative2:
			gram = [];
			a = np.zeros((1, 100), dtype=float)
			for i in range(len(pep) - 3):
				gram.append(pep[i:i + 3])
			gram.append(pep[-3:])
			
			for x in gram:
				for i, num in features.items():
					if i == x:
						a += num
			final_vec2.append(a.tolist()[0])


	elif mode=='one-hot':
		aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
		for n in T_positive2:
			vec=[]
			for a in n:
				for x in aa:
					if a==x:
						if x!='X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			final_vec1.append(vec)

		for n in T_negative2:
			vec=[]
			for a in n:
				for x in aa:
					if a==x:
						if x!='X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			final_vec2.append(vec)


	elif mode=='one-hot_structure':
		aa=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','X']
		structure=['C','H','E','X'];solvent=['X','E','M','B'];order=['X','*','.']
		for n in T_positive2:
			vec=[]
			for a in n:
				for x in aa:
					if a==x:
						if x!='X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			#final_vec3.append(vec)
			####基于二级结构
			for i in positive1[n]:
				for x in structure:
					if i==x:
						if x!='X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			####基于疏水性
			for i in positive2[n]:
				for x in solvent:
					if i==x:
						if x!='X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			####基于有序性
			for i in positive3[n]:
				for x in order:
					if i==x:
						if x!='X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			####基于KNN
			vec.append(KNN[n])
			final_vec1.append(vec)

		for n in T_negative2:
			vec = []
			for a in n:
				for x in aa:
					if a == x:
						if x != 'X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			# final_vec4.append(vec)
			####基于二级结构
			for i in all_dataset1[n]:
				for x in structure:
					if i == x:
						if x != 'X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			####基于疏水性
			for i in all_dataset2[n]:
				for x in solvent:
					if i == x:
						if x != 'X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			####基于有序性
			for i in all_dataset3[n]:
				for x in order:
					if i == x:
						if x != 'X':
							vec.append(1)
						else:
							vec.append(0.5)
					else:
						vec.append(0)
			####基于KNN
			vec.append(KNN[n])
			final_vec2.append(vec)

	vec = np.array(final_vec1+ final_vec2)
	vec_label = np.array([1] * len(final_vec1) + [0] * len(final_vec2))

	return vec,vec_label

	# if algorithm=='svm':
	# 	clf=svm.LinearSVC()
	# 	scores = model_selection.cross_val_score(clf, vec, y=vec_label, cv=5)
	# 	print('Per accuracy in 5-fold CV:')
	# 	print(scores)
	# 	print("Accuracy of svm: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	#
	# elif algorithm=='mlp':
	# 	# train_data, test_data, train_labels, test_lables = train_test_split(vec, vec_label, test_size=0.25, random_state=33)
	# 	# classifier = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(175, 75), random_state=1)
	# 	# classifier.fit(train_data, train_labels)
	# 	# print classifier.score(test_data,test_lables)
	#
	# 	classifier = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(175, 75), random_state=1)
	# 	scores = model_selection.cross_val_score(classifier, vec, y=vec_label, cv=5)
	# 	print('Per accuracy in 5-fold CV:')
	# 	print(scores)
	# 	print("Accuracy of mlp: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def mul_ROC(fun,*para):
	print para
	for item in para:
		print item
		vec1,vec_label1=fun(item)
		cv = StratifiedKFold(n_splits=10)
		if item[0][4]=='svm':
			classifier = svm.SVC(kernel='linear', probability=True)
		else:
			classifier = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(175, 75), random_state=1)
	
		mean_tpr = 0.0
		mean_fpr = np.linspace(0, 1, 100)
		all_tpr = []
		cnt = 0
		for i, (train, test) in enumerate(cv.split(vec1, vec_label1)):
			cnt = cnt + 1
			probas_ = classifier.fit(vec1[train], vec_label1[train]).predict_proba(vec1[test])
			fpr, tpr, thresholds = roc_curve(vec_label1[test], probas_[:, 1])
			mean_tpr += interp(mean_fpr, fpr, tpr)
			mean_tpr[0] = 0.0

		# Plot ROC curve.
		mean_tpr /= cnt
		mean_tpr[-1] = 1.0
		mean_auc = auc(mean_fpr, mean_tpr)
		plt.plot(mean_fpr, mean_tpr, '-', label='ROC %s (area = %0.2f)' % (item[2]+'&'+item[4],mean_auc), lw=2)

	plt.xlim([0, 1.0])
	plt.title('ROC of PKC kinase')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend(loc="lower right")
	plt.show()


if __name__=='__main__':
	paras=[r'C:\Users\xzw\Desktop\human_phosphoplus.csv',r'E:\python_project\model\non_overlap.model',r'E:\python_project\model\protein.model',r'C:\Users\xzw\Desktop\ATM_merge.fasta',
		   r'C:\Users\xzw\Desktop\CDK_merge.fasta',r'C:\Users\xzw\Desktop\CK2_merge.fasta',r'C:\Users\xzw\Desktop\PKC_merge.fasta',r'C:\Users\xzw\Desktop\PKA_merge.fasta']
	mul_ROC(extract,[paras[6], paras[0],'no_overlap',paras[1],'svm'],
					[paras[6], paras[0], 'one-hot_structure','one-hot_structure', 'svm'],
					[paras[6], paras[0], 'no_overlap',paras[1], 'mlp'],
					[paras[6], paras[0], 'overlap','overlap', 'mlp'],
					[paras[6], paras[0], 'our_overlap',paras[2], 'mlp'],
					[paras[6], paras[0], 'one-hot','one-hot', 'mlp'],
					[paras[6], paras[0], 'one-hot_structure','one-hot_structure', 'mlp'],
					[paras[6], paras[0], 'overlap', 'overlap','svm'],
					[paras[6], paras[0], 'our_overlap', paras[2], 'svm'],
					[paras[6], paras[0], 'one-hot', 'one-hot', 'svm'],
			)

	
