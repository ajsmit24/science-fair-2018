
import csv


atomNumb=100
numbIso=50
numbElem=150
nue=0
row=8
col=1
hold=0
ans=""

for p in range(0,numbElem):
	atomNumb+=p
	if (col==2 and hold!=17):
		hold=hold+1
	elif(col==18):
		col=1
		row=row+1
	else:
		col=col+1
		
	
	
	for n in range(0,numbIso):
		nue=atomNumb+n
		ans=str(atomNumb)+','+str(nue)+','+str(atomNumb)+','+"--"+','+"--"+','+str(col)+','+str(row)+','+"--"+','+"--"+"|"
		with open('ans.csv', 'w') as csvfile:
			fieldnames = ['first_name', 'last_name']
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerow({'protons': str(atomNumb), 'nuetrons': str(nue),'electrons': str(atomNumb),'NueConfigNumb': "--",'ProtConfigNumb': "--",'periodic x': str(col),'periodic y': str(row),'ProtElectConfig': "--",'NueConfig': "--"})
	for n in range(0,numbIso):
		nue=atomNumb-n
		ans=str(atomNumb)+','+str(nue)+','+str(atomNumb)+','+"--"+','+"--"+','+str(col)+','+row+','+"--"+','+"--"+"|"
		with open('ans.csv', 'w') as csvfile:
			fieldnames = ['first_name', 'last_name']
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writeheader()
			writer.writerow({'protons': str(atomNumb), 'nuetrons': str(nue),'electrons': str(atomNumb),'NueConfigNumb': "--",'ProtConfigNumb': "--",'periodic x': str(col),'periodic y': str(row),'ProtElectConfig': "--",'NueConfig': "--"})
		