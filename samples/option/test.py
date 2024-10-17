from blackscholes import BlackScholesCall
call = BlackScholesCall(S=1, K=0.95, T=5, r=0.01, sigma=0.15, q=0)
call_price = call.price()  
print(call_price)
call.delta() 
call.spot_delta()
call.charm() 