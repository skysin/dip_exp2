with open('output.txt', 'r') as f:
	a = f.readline().split(' ')
	# print(a)
	for i in range(256):
		sum = 0
		for j in range(10):
			sum += float(a[i + j * 256])
			# print(float(a[i + j * 256]))
		print(sum)