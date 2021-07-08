n=int(input())
ans=set()
for i in range(1,int(n**0.5+3)):
    if n%i==0:
        ans.add(tuple(sorted([i,n//i])))
print(sorted(ans))