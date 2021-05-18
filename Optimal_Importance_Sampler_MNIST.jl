using  Statistics, StatsBase, Distributions,DataFrames, CSV, Plots, Match
pyplot()

# Helper functions:
include("routines.jl")

## TEST SET CONSTRUCTION AND ANALYSIS
df=CSV.read("fold0.txt",DataFrame;delim=" ")
Ntest=size(df,1)

p1_pred = df."prob"
c_pred = 1*(p1_pred .> 0.5)
c_true = 1*(df."label" .> 0.5)


global true_positive=sum((c_pred.==1) .& (c_true.==1))
global false_positive=sum((c_pred.==1) .& (c_true.==0))
global false_negative=sum((c_pred.==0) .& (c_true.==1))
global true_negative=sum((c_pred.==0) .& (c_true.==0))
global F1Test_true=true_positive/(true_positive + 0.5*(false_positive+false_negative))    
global SpecTest_true=true_negative/(true_negative + false_positive)

lambda=0.9
p1_true_guess = lambda*p1_pred .+ (1-lambda)*0.5 # guess of the true label on test set

# Define the parameters for the Optimal Sampling Distribution:
htilde = happroxFn(c_pred,p1_true_guess;metric="F",metric_par=0.5)

Mvals=Int.(ceil.(LinRange(10,10*Ntest,100)))

percent=zeros(length(Mvals))

spec_mean=zeros(length(Mvals))
spec_lower=zeros(length(Mvals))
spec_upper=zeros(length(Mvals))
acc_mean=zeros(length(Mvals))
acc_lower=zeros(length(Mvals))
acc_upper=zeros(length(Mvals))
precision_mean=zeros(length(Mvals))
precision_lower=zeros(length(Mvals))
precision_upper=zeros(length(Mvals))
recall_mean=zeros(length(Mvals))
recall_lower=zeros(length(Mvals))
recall_upper=zeros(length(Mvals))
f1_mean=zeros(length(Mvals))
f1_lower=zeros(length(Mvals))
f1_upper=zeros(length(Mvals))


Unlabelled=1:Ntest

# Decide which points to sample:
q=htilde/sum(htilde) # importance distribution
dist=Categorical(q)   

for experiment=1:length(Mvals)

    local Nlabel=Mvals[experiment] # number of points we will sample from the unlabelled set and ask to be labelled
    local ToLabel=rand(dist,Nlabel) # sample Nlabel points from the importance distribution
    local total_samples_used = length(unique(ToLabel)) # number of unique samples
    
    Fhat_acc_mean, Fhat_acc_var = FhatIS(c_pred[ToLabel],c_true[ToLabel],q[ToLabel];metric="Accuracy")
    
    F1hat_mean, F1hat_var = FhatIS(c_pred[ToLabel],c_true[ToLabel],q[ToLabel];metric="F",metric_par=0.5)
    
    Fhat_precision_mean, Fhat_precision_var = FhatIS(c_pred[ToLabel],c_true[ToLabel],q[ToLabel];metric="F",metric_par=1)

    Fhat_recall_mean,Fhat_recall_var = FhatIS(c_pred[ToLabel],c_true[ToLabel],q[ToLabel];metric="F",metric_par=0) 

    Fhat_spec_mean,Fhat_spec_var = FhatIS(c_pred[ToLabel],c_true[ToLabel],q[ToLabel];metric="Specificity") 
    
    # convert this to confidence bounds by fitting this to say a beta distribution:
    
    global confidence_level=0.9 # one standard deviation (for a Gaussian)
    
    println("-----------------------------------------")
    println(experiment)
    
    println("total samples used  = $(total_samples_used)")
    println("% of test data      = $(total_samples_used/Ntest)")

    percent[experiment]=total_samples_used/Ntest
    
    println("true accuracy       = $(mean(c_true .== (p1_pred .> 0.5)))")
    local mn,lower,upper = confidenceBeta(Fhat_acc_mean,Fhat_acc_var,confidence_level)
    println("estimated accuracy  = $mn [$lower,$upper]")
        
    acc_mean[experiment]=mn
    acc_lower[experiment]=lower
    acc_upper[experiment]=upper
    
    println("")
        
    println("true precision      = $(true_positive/(true_positive+false_positive))")
    local mn,lower,upper = confidenceBeta(Fhat_precision_mean,Fhat_precision_var,confidence_level)
    println("estimated precision = $mn [$lower,$upper]")

    precision_mean[experiment]=Fhat_precision_mean
    precision_lower[experiment]=lower
    precision_upper[experiment]=upper

    println("")
    
    println("true recall         = $(true_positive/(true_positive+false_negative))")
    local mn,lower,upper = confidenceBeta(Fhat_recall_mean,Fhat_recall_var,confidence_level)
    println("estimated recall    = $mn [$lower,$upper]")

    recall_mean[experiment]=Fhat_recall_mean
    recall_lower[experiment]=lower
    recall_upper[experiment]=upper

    println("")
    
    println("true F1             = $(F1Test_true)")
    local mn,lower,upper = confidenceBeta(F1hat_mean,F1hat_var,confidence_level)
    println("estimated F1        = $mn [$lower,$upper]")
    
    f1_mean[experiment]=F1hat_mean
    f1_lower[experiment]=lower
    f1_upper[experiment]=upper
    
    println("")

    println("true specificity             = $(SpecTest_true)")
    local mn,lower,upper = confidenceBeta(Fhat_spec_mean,Fhat_spec_var,confidence_level)
    println("estimated specificity        = $mn [$lower,$upper]")
    
    spec_mean[experiment]=Fhat_spec_mean
    spec_lower[experiment]=lower
    spec_upper[experiment]=upper
    
end

mp=maximum(percent)

acc_true = (mean(c_true .== (p1_pred .> 0.5)))
ribbon_upper=acc_upper - acc_mean
ribbon_lower=acc_mean - acc_lower
plot(reuse=false,show=true)
plot_acc=plot!(percent,acc_mean,ribbon=(ribbon_lower,ribbon_upper),label="$(confidence_level) confidence interval",framestyle = :box,minorgrid=true)
plot!([0,mp],[acc_true,acc_true],label="true value",legend=:bottomright,ylims=[0.9*acc_true,1.1*acc_true])
title!("Accuracy")
xlabel!("fraction of test data labelled")

precision_true = true_positive/(true_positive+false_positive)
ribbon_upper=precision_upper - precision_mean
ribbon_lower=precision_mean - precision_lower
plot(reuse=false,show=true)
plot_precision=plot!(percent,precision_mean,ribbon=(ribbon_lower,ribbon_upper),label="$(confidence_level) confidence interval",framestyle = :box,minorgrid=true)
plot!(plot_precision)
plot!([0,mp],[precision_true,precision_true],label="true value",legend=:bottomright,ylims=[0.9*precision_true,1.1*precision_true])
title!("Precision")
xlabel!("fraction of test data labelled")

recall_true = true_positive/(true_positive+false_negative)
ribbon_upper=recall_upper - recall_mean
ribbon_lower=recall_mean - recall_lower
plot(reuse=false,show=true)
plot_recall=plot!(percent,recall_mean,ribbon=(ribbon_lower,ribbon_upper),label="$(confidence_level) confidence interval",framestyle = :box,minorgrid=true)
plot!(plot_recall)
plot!([0,mp],[recall_true,recall_true],label="true value",legend=:bottomright,ylims=[0.9*recall_true,1.1*recall_true])
title!("Recall")
xlabel!("fraction of test data labelled")

f1_true = F1Test_true
ribbon_upper=f1_upper - f1_mean
ribbon_lower=f1_mean - f1_lower
plot(reuse=false,show=true)
plot_f1=plot!(percent,f1_mean,ribbon=(ribbon_lower,ribbon_upper),label="$(confidence_level) confidence interval",framestyle = :box,minorgrid=true)
plot!(plot_f1)
plot!([0,mp],[f1_true,f1_true],label="true value",legend=:bottomright,ylims=[0.9*f1_true,1.1*f1_true])
title!("F1")
xlabel!("fraction of test data labelled")

spec_true = SpecTest_true
ribbon_upper=spec_upper - spec_mean
ribbon_lower=spec_mean - spec_lower
plot(reuse=false,show=true)
plot_spec=plot!(percent,spec_mean,ribbon=(ribbon_lower,ribbon_upper),label="$(confidence_level) confidence interval",framestyle = :box,minorgrid=true)
plot!(plot_spec)
plot!([0,mp],[spec_true,spec_true],label="true value",legend=:bottomright,ylims=[0.9*spec_true,1.1*spec_true])
title!("Specificity")
xlabel!("fraction of test data labelled")
