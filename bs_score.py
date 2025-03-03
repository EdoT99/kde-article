import math


b = [100,400,900]
c = [100,1600,8100]
a = [100,100,100]
d = [100,6400,72900]
y = [26,10,10,10,8]
sets = [a,b,c,d,y]
for dat in sets:
        suma = sum(dat)
        base = len(dat)
        prob = [el/suma for el in dat]
        entropy = - sum([(p*(math.log(p,base))) for p in prob ])
        balance_score = 1-entropy
        print(round(balance_score,3))