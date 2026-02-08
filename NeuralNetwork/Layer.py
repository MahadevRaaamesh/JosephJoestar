input=[1,2,3]
weight=[[2,3,4,5],[7,5,4,1],[1,3,1,9],[2,2,2,2]]
bias=[2,3,4,2]
op=[]
for nweight,nbias in zip(weight,bias):
    nop=0
    for nip,weight in zip(input,nweight):
        nop+=nip*weight
    nop+=nbias
    op.append(nop)    
print(op)    
