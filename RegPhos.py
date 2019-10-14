#coding=utf8


##对regphos文件的处理，再次传到git上

with open(r'E:\python_project\original_data\RegPhos.fasta')as f:
    lst=f.readlines()[1:];sub=[];x=0
    for i in lst:
        if i.split('\t')[4]:
            x+=1
            # print i.split('\t')[4]
            if i.split('\t')[1] not in sub:
                sub.append(i.split('\t')[1])
    print x;print len(sub)
f.close()