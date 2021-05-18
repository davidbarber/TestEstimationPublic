using  Statistics,StatsBase,Distributions,DataFrames,CSV,Plots,Match,Zygote
pyplot()

# Helper functions:
include("routines.jl")

## TEST SET CONSTRUCTION AND ANALYSIS
df=CSV.read("JigsawPred.csv",DataFrame;delim=",",header=false)

dfm=Matrix(df)
Ntest=size(df,1)
c_pred=zeros(Int,Ntest,6) # multi-label problem with 6 labels
for n=1:Ntest
    for k = 1:6
        if dfm[n,k]>0.5
            c_pred[n,k]=1
        end
    end
end
p1_pred = 1.0*dfm # for this classifier, the predictions are essentially deterministic


df2=CSV.read("JigsawTrue.csv",DataFrame;delim=",",header=false)
dfm=Matrix(df2)
c_true=zeros(Int,Ntest,6) # multi-label problem with 6 labels
for n=1:Ntest
    for k = 1:6
        if dfm[n,k]>0.5
            c_true[n,k]=1
        end
    end
end

F1Test_micro_true,F1Test_macro_true,accuracy_individual,accuracy_joint = MultiMetrics(c_pred,c_true)    

# Bernoulli Sampling approximation:
lambda=0.9
p1_true_guess = lambda*p1_pred .+ (1-lambda)*0.5 # guess of the true label on test set

# Define the parameters for the Optimal Sampling Distribution:
htilde,dummy = happroxFn_multi_micro(c_pred,p1_true_guess)

Mvals=LinRange(10,Ntest,10) # expected number of samples

percent=zeros(length(Mvals))

acc_mean=zeros(length(Mvals))
acc_lower=zeros(length(Mvals))
acc_upper=zeros(length(Mvals))

f1_micro_mean=zeros(length(Mvals))
f1_micro_lower=zeros(length(Mvals))
f1_micro_upper=zeros(length(Mvals))

f1_macro_mean=zeros(length(Mvals))
f1_macro_lower=zeros(length(Mvals))
f1_macro_upper=zeros(length(Mvals))


for experiment=1:length(Mvals)
        
    global M = Mvals[experiment] # average number of elements in each test set replicate        
    global b = optimalb(htilde,M)
    global s = (rand(Ntest) .< b)
    global total_samples_used = sum(s)

    ToLabel=findall(s.==1)

    F1hat_micro_mean,F1hat_micro_var=FhatBS_micro(c_pred[ToLabel,:],c_true[ToLabel,:],b[ToLabel])
    F1hat_macro_mean,F1hat_macro_var=FhatBS_macro(c_pred[ToLabel,:],c_true[ToLabel,:],b[ToLabel])
        
    # convert this to confidence bounds by fitting this to say a beta distribution:
    
    global confidence_level=0.9 
    
    println("-----------------------------------------")
    println("total samples used  = $total_samples_used")
    println("% of test data      = $(total_samples_used/Ntest)")
    percent[experiment]=total_samples_used/Ntest
    
    println("")

    println("true F1             = $(F1Test_micro_true)")
    local mn,lower,upper = confidenceBeta(F1hat_micro_mean,F1hat_micro_var,confidence_level)
    println("estimated F1        = $mn [$lower,$upper]")
    
    f1_micro_mean[experiment]=F1hat_micro_mean
    f1_micro_lower[experiment]=lower
    f1_micro_upper[experiment]=upper
    
    println("")
    
    println("true F1             = $(F1Test_macro_true)")
    local mn,lower,upper = confidenceBeta(F1hat_macro_mean,F1hat_macro_var,confidence_level)
    println("estimated F1        = $mn [$lower,$upper]")
    
    f1_macro_mean[experiment]=F1hat_macro_mean
    f1_macro_lower[experiment]=lower
    f1_macro_upper[experiment]=upper

end

mp=maximum(percent)

f1_true = F1Test_micro_true
ribbon_upper=f1_micro_upper - f1_micro_mean
ribbon_lower=f1_micro_mean - f1_micro_lower
plot(reuse=false,show=true)
plot_f1_micro=plot!(percent,f1_micro_mean,ribbon=(ribbon_lower,ribbon_upper),label="$(confidence_level) confidence interval",framestyle = :box,minorgrid=true)
plot!(plot_f1_micro)
plot!([0,mp],[f1_true,f1_true],label="true value",legend=:bottomright)#,ylims=[0.5*f1_true,1.5*f1_true])
title!("F1 Micro")
xlabel!("fraction of test data labelled")

f1_true = F1Test_macro_true
ribbon_upper=f1_macro_upper - f1_macro_mean
ribbon_lower=f1_macro_mean - f1_macro_lower
plot(reuse=false,show=true)
plot_f1_macro=plot!(percent,f1_macro_mean,ribbon=(ribbon_lower,ribbon_upper),label="$(confidence_level) confidence interval",framestyle = :box,minorgrid=true)
plot!(plot_f1_macro)
plot!([0,mp],[f1_true,f1_true],label="true value",legend=:bottomright)#,ylims=[0.5*f1_true,1.5*f1_true])
title!("F1 Macro")
xlabel!("fraction of test data labelled")
