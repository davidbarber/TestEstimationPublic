MissingFloat=Union{Missing,Float64}

function mean1(x)
    out=zeros(size(x,2))
    for i=1:size(x,2)
        out[i]=mean(skipmissing(x[:,i]))
    end
    return out
end

function std1(x)
    out=zeros(size(x,2))
    for i=1:size(x,2)
        out[i]=std(skipmissing(x[:,i]))
    end
    return out
end

function mean2(x)
    out=zeros(size(x,1))
    for i=1:size(x,1)
        out[i]=mean(skipmissing(x[i,:]))
    end
    return out
end

small_value = 1e-12

function F1(N)
    # N[i,j] is the number of times model predicts class i and true is class j
    # I use class 1 for the "positive" class and class 2 for the "negative" class
    true_positive = N[1,1]
    false_positive = N[1,2]
    false_negative = N[2,1]
    return true_positive/(true_positive + 0.5*(false_positive+false_negative))    
end

function Precision(N)
    # N[i,j] is the number of times model predicts class i and true is class j
    # I use class 1 for the "positive" class and class 2 for the "negative" class
    true_positive = N[1,1]
    false_positive = N[1,2]
    false_negative = N[2,1]
    return true_positive/(true_positive + false_positive)
end

function Recall(N)
    # N[i,j] is the number of times model predicts class i and true is class j
    # I use class 1 for the "positive" class and class 2 for the "negative" class
    true_positive = N[1,1]
    false_positive = N[1,2]
    false_negative = N[2,1]
    return true_positive/(true_positive + false_negative)
end

function fitBeta(mean_val,var_val)
    mn=mean_val
    s2=min(var_val,0.5^2)
    a=(1-mn)*mn^2/s2 - mn
    b=a*(1/mn-1)
    return a,b
end

function confidenceBeta(mean_fhat,var_fhat,confidence)
#    println([mean_fhat,var_fhat,confidence])
    if ismissing(mean_fhat)
        return missing,missing,missing
    end
    if var_fhat<1e-20
        return mean_fhat,mean_fhat,mean_fhat
    end
    var_fhat = var_fhat + small_value # try to avoid numerical issues
    
    tail=0.5*(1-confidence) # tail probabilities (for each tail)
   
    a,b = fitBeta(mean_fhat,var_fhat)

    a = max(a,0.001) # ugly hack to avoid problems
    b = max(b,0.001) # ugly hack to avoid problems

    dist=Beta(a,b)
    mn = a/(a+b) # mean posterior error

    upper_limit = quantile(dist,confidence+tail)
    lower_limit = quantile(dist,tail)

    return mean_fhat, lower_limit, upper_limit
end

function f(c_pred,c_true;metric,metric_par=0)
    @match metric begin
        "F" =>  return  (c_pred .==1) .* (c_true .==1)
        "Accuracy" => return  (c_pred .==1) .* (c_true .==1) + (c_pred .==0) .* (c_true .==0)
        "Specificity" => return  (c_pred .==0) .* (c_true .==0)
    end
end

function g(c_pred,c_true;metric,metric_par=0)
    alpha=metric_par
    @match metric begin
        "F" =>  return alpha.*(c_pred .==1) + (1 - alpha) .* (c_true .== 1)
        "Accuracy" => return ones(size(c_pred))
        "Specificity" => return (c_true .==0)
    end
end

micro_add=1e-10

function f_micro(c_pred,c_true)
    #return  sum((c_pred .==1) .* (c_true .==1),dims=2)
    return  (c_pred .==1) .* (c_true .==1) .+ micro_add
end

function g_micro(c_pred,c_true)
    alpha=0.5
    #return sum(alpha.*(c_pred .==1) + (1 - alpha) .* (c_true .== 1),dims=2)
    return alpha.*(c_pred .==1) + (1 - alpha) .* (c_true .== 1) .+ micro_add
end


function fgtilde(c_pred::Vector{Int};metric,metric_par=0)
    ftilde1 = f(c_pred,ones(size(c_pred));metric,metric_par)
    gtilde1 = g(c_pred,ones(size(c_pred));metric,metric_par)
    ftilde0 = f(c_pred,zeros(size(c_pred));metric,metric_par)
    gtilde0 = g(c_pred,zeros(size(c_pred));metric,metric_par)
    return (ftilde1,ftilde0,gtilde1,gtilde0)
end

function FhatIS(c_pred::Vector{Int},c_true::Vector{Int},q;metric,metric_par=0)
    ftilde = f(c_pred,c_true;metric,metric_par)
    gtilde = g(c_pred,c_true;metric,metric_par)
    Fapprox = (small_value + sum(ftilde./q))./(small_value + sum(gtilde./q))
    mn = CheckMissing(Fapprox)

    M = length(c_pred) # number of samples
    (ftilde1,ftilde0,gtilde1,gtilde0) = fgtilde(c_pred;metric,metric_par)    

    lambda=0.999 # NEED TO CHECK THIS!!!!
    c_true_s=lambda*c_true .+ (1-lambda)*0.5
    gtilde = c_true_s.*gtilde1 + (1 .- c_true_s).*gtilde0
    htilde2 = c_true_s.*(ftilde1 -Fapprox.*gtilde1).^2 + (1 .- c_true_s).*(ftilde0 .- Fapprox.*gtilde0).^2

    yhat = mean( gtilde./q)/Ntest
    numhat = mean( htilde2./(q.*q))/Ntest
    
    var = (small_value.^2 .+ numhat) ./ (small_value.^2 .+  yhat^2)/ (M*Ntest) 
    
    if ismissing(mn) | ismissing(var)
        return missing,missing
    else
        return mn,var
    end
    
end

function FhatBS(c_pred::Vector{Int},c_true::Vector{Int},b;metric,metric_par=0)
    ftilde = f(c_pred,c_true;metric,metric_par)
    gtilde = g(c_pred,c_true;metric,metric_par)
    Fapprox = (small_value + sum(ftilde./b))./(small_value + sum(gtilde./b))
    mn  = CheckMissing(Fapprox)

    (ftilde1,ftilde0,gtilde1,gtilde0) = fgtilde(c_pred;metric,metric_par)

    lambda=0.999 # slightly smooth to avoid numerical issues
    c_true_s=lambda*c_true .+ (1-lambda)*0.5
    gtilde = c_true_s.*gtilde1 + (1 .- c_true_s).*gtilde0
    htilde2 = c_true_s.*(ftilde1 -Fapprox.*gtilde1).^2 + (1 .- c_true_s).*(ftilde0 .- Fapprox.*gtilde0).^2
    htilde = c_true_s.*(ftilde1 -Fapprox.*gtilde1) + (1 .- c_true_s).*(ftilde0 .- Fapprox.*gtilde0)
   
    var = sum( (htilde2./b .- htilde.^2)./b) / (sum(gtilde./b)).^2

    if ismissing(mn) | ismissing(var)
        return missing,missing
    else
        return mn,var
    end
end


function FhatBS_micro(c_pred,c_true,b)
    ftilde = sum(f_micro(c_pred,c_true),dims=2)
    gtilde = sum(g_micro(c_pred,c_true),dims=2)
    Fapprox = (small_value + sum(ftilde./b))./(small_value + sum(gtilde./b))
    mn  = CheckMissing(Fapprox)

    lambda=0.999
    c_true_s=lambda*c_true .+ (1-lambda)*0.5

    ftilde1 = sum(f_micro(c_pred,ones(size(c_true))),dims=2)
    ftilde0 = sum(f_micro(c_pred,zeros(size(c_true))),dims=2)
    gtilde1 = sum(g_micro(c_pred,ones(size(c_true))),dims=2)
    gtilde0 = sum(g_micro(c_pred,zeros(size(c_true))),dims=2)

    htilde2 = c_true_s.*(ftilde1 -Fapprox.*gtilde1).^2 + (1 .- c_true_s).*(ftilde0 .- Fapprox.*gtilde0).^2
    htilde = c_true_s.*(ftilde1 -Fapprox.*gtilde1) + (1 .- c_true_s).*(ftilde0 .- Fapprox.*gtilde0)
   
    gtilde = c_true.*gtilde1 + (1 .- c_true).*gtilde0
    
    var = sum( (htilde2./b .- htilde.*htilde)./b) / (sum(gtilde./b)).^2
    
    if ismissing(mn) | ismissing(var)
        return missing,missing
    else
        return mn,var
    end
end


function FhatBS_macro(c_pred,c_true,b)

    den = length(b)
    
    a_bar = vec(sum(  ((c_pred.==1) .& (c_true.==1))./b, dims=1)./den) 
    b_bar = vec(sum(  ((c_pred.==1) .& (c_true.==0))./b, dims=1)./den) 
    c_bar = vec(sum(  ((c_pred.==0) .& (c_true.==1))./b, dims=1)./den) 
        
    Fapprox = Fabc(a_bar,b_bar,c_bar)
    mn  = CheckMissing(Fapprox)

    Ga = gradient(a -> Fabc(a,b_bar,c_bar), a_bar)[1]
    Gb = gradient(b -> Fabc(a_bar,b,c_bar), b_bar)[1]
    Gc = gradient(c -> Fabc(a_bar,b_bar,c), c_bar)[1]
    
    Ga2= Ga.*Ga 
    Gb2= Gb.*Gb
    Gc2= Gc.*Gc

    Ntest=size(c_pred,1)
    vA=zeros(size(c_pred,2))
    vB=zeros(size(c_pred,2))
    vC=zeros(size(c_pred,2))
    lambda=0.9
    c_pred_s =lambda*c_pred .+ (1-lambda)*0.5 
    c_true_s =lambda*c_true .+ (1-lambda)*0.5 # this isn't really correct since we need to include additional terms
    c_true_s =1.0*c_true
    for k=1:size(c_pred,2)
        for n=1:size(c_pred,1)            
            vA[k] += c_pred_s[n,k]*c_true_s[n,k]*Ga2[k]*(1 ./ b[n] -1)./b[n]
            vB[k] += c_pred_s[n,k]*(1-c_true_s[n,k])*Gb2[k]*(1 ./ b[n] -1)./b[n]
            vC[k] += (1-c_pred_s[n,k])*c_true_s[n,k]*Gc2[k]*(1 ./ b[n] -1)./b[n]
        end
        # Note that there is no term E[DeltaA*DeltaB] since for a given c_true this term is zero
    end
    vA=vA/den^2
    vB=vB/den^2
    vC=vC/den^2

    var = CheckMissing(sum(vA+vB+vC)) 

    if ismissing(mn) | ismissing(var)
        return missing,missing
    else
        return mn,var
    end
    
end


function Fabc(a::Vector,b::Vector,c::Vector)
    C=length(a)
    aa=a .+ small_value
    bb=b .+ small_value
    cc=c .+ small_value
    
    p= sum(aa./(aa + bb))/C
    r= sum(aa./(aa + cc))/C
    
    return 2*p*r/(p+r)
end


function FapproxFn(c_pred::Vector{Int},p1_true_guess;metric,metric_par=0)
    (ftilde1,ftilde0,gtilde1,gtilde0) = fgtilde(c_pred;metric,metric_par)    
    ftilde = p1_true_guess.*ftilde1 + (1 .- p1_true_guess).*ftilde0
    gtilde = p1_true_guess.*gtilde1 + (1 .- p1_true_guess).*gtilde0
    return sum(ftilde)/sum(gtilde)
end

function happroxFn(c_pred::Vector{Int},p1_true_guess;metric,metric_par=0)
    Fapprox = FapproxFn(c_pred,p1_true_guess;metric,metric_par)
    (ftilde1,ftilde0,gtilde1,gtilde0) = fgtilde(c_pred;metric,metric_par)    
    htilde2 = p1_true_guess.*(ftilde1 -Fapprox.*gtilde1).^2 + (1 .- p1_true_guess).*(ftilde0 .- Fapprox.*gtilde0).^2
    return sqrt.(htilde2)
end


function MultiMetrics(c_pred,c_true)    
    true_positive=sum((c_pred.==1) .& (c_true.==1))
    false_positive=sum((c_pred.==1) .& (c_true.==0))
    false_negative=sum((c_pred.==0) .& (c_true.==1))
    true_negative=sum((c_pred.==0) .& (c_true.==0))
    F1_micro=true_positive/(true_positive + 0.5*(false_positive+false_negative)) # micro F1

    accuracy_individual=(true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative) # fraction of individually correct entries
    accuracy_joint=mean(all(c_true .== c_pred,dims=2)) # fraction of correct vector predictions

    true_positive=mean((c_pred.==1) .& (c_true.==1),dims=1)
    false_positive=mean((c_pred.==1) .& (c_true.==0),dims=1)
    false_negative=mean((c_pred.==0) .& (c_true.==1),dims=1)
    true_negative=mean((c_pred.==0) .& (c_true.==0),dims=1)
    Prec = mean(true_positive./(true_positive .+ false_positive))
    Rec = mean(true_positive./(true_positive .+ false_negative))
    
    F1_macro=2*Prec*Rec/(Prec+Rec)

    return F1_micro,F1_macro,accuracy_individual,accuracy_joint
end


function FapproxFn_multi_micro(c_pred,p1_true_guess)
    alpha=0.5
    C=size(c_pred,2)
    num=0.0
    den=0.0
    for n=1:size(c_pred,1)
        for k=1:C
            num=num+ c_pred[n,k]*p1_true_guess[n,k]
            den=den+ alpha*c_pred[n,k] + (1-alpha)*p1_true_guess[n,k]
        end
    end
    return num/den
end
        

function h_multi_micro(cp,ct,F)    
    return f_micro(cp,ct)-F.*g_micro(cp,ct)
end


function happroxFn_multi_micro(c_pred,p1_true_guess)
    F =FapproxFn_multi_micro(c_pred,p1_true_guess)
    p1=1.0*p1_true_guess
    p0=1.0 .- p1
    h2=zeros(size(c_pred,1))
    avh=zeros(size(c_pred,1))
    for n=1:size(c_pred,1)
        tmp1=0
        for k=1:size(c_pred,2)
            tmp1 = tmp1 + p0[n,k]*h_multi_micro(c_pred[n,k],0,F)^2 +
                p1[n,k]*h_multi_micro(c_pred[n,k],1,F)^2                            
        end
        tmp2=0
        for k=1:size(c_pred,2)
            tmp2 = tmp2 + (p0[n,k]*h_multi_micro(c_pred[n,k],0,F) +
                p1[n,k]*h_multi_micro(c_pred[n,k],1,F))^2
        end
        tmp3=0
        for k=1:size(c_pred,2)
            tmp3 = tmp3 + p0[n,k]*h_multi_micro(c_pred[n,k],0,F) +
                    p1[n,k]*h_multi_micro(c_pred[n,k],1,F)
        end
        h2[n]=tmp1-tmp2+tmp3^2

        for k=1:size(c_pred,2)
            avh[n] += p0[n,k]*h_multi_micro(c_pred[n,k],0,F) +
                p1[n,k]*h_multi_micro(c_pred[n,k],1,F)
        end
        
    end
    return sqrt.(h2), avh
end


function happroxFn_multi_macro(c_pred,p1_true_guess)
       
    a_bar = vec(mean( p1_true_guess .* ((c_pred.==1) .& (c_true.==1)), dims=1)) # check!
    b_bar = vec(mean( (1 .- p1_true_guess) .* ((c_pred.==1) .& (c_true.==0)), dims=1)) # check!
    c_bar = vec(mean( p1_true_guess .* ((c_pred.==0) .& (c_true.==1)), dims=1)) # check! 
        
    Ga = gradient(a -> Fabc(a,b_bar,c_bar), a_bar)[1]
    Gb = gradient(b -> Fabc(a_bar,b,c_bar), b_bar)[1]
    Gc = gradient(c -> Fabc(a_bar,b_bar,c), c_bar)[1]
    
    Ga2= Ga.*Ga 
    Gb2= Gb.*Gb
    Gc2= Gc.*Gc
        
    p1=1.0*p1_true_guess
    p0=1.0 .- p1
    h2=zeros(size(c_pred,1))
    for n=1:size(c_pred,1)
        tmpa=0
        for k=1:size(c_pred,2)
            tmpa = tmpa + c_pred[n,k]*p1[n,k]*Ga2[k]
        end
        tmpb=0
        for k=1:size(c_pred,2)
            tmpb = tmpb + c_pred[n,k]*p0[n,k]*Gb2[k]
        end
        tmpc=0
        for k=1:size(c_pred,2)
            tmpc = tmpc + (1-c_pred[n,k])*p1[n,k]*Gc2[k] #  check!
        end

        h2[n]=tmpa+tmpb+tmpc
    end
    return sqrt.(h2)
end



function CheckMissing(x)
    if isnan(x) | ismissing(x)
        return missing
    else
        return x
    end
end


function optimalb(h,M)
    totalh=sum(h)
    sumind=0
    order=sortperm(-h)
    b=zeros(length(h))
    if M*h[order[1]]./totalh <=1 
        return M*h/totalh
    end
    
    count=0
    sumh=0
    for ind in order
        count += 1
        sumh += h[ind]
        D=totalh - sumh
        N=M-count
        G=h[ind]*N/D
        if G <=1
            b[order[1:count-1]]=ones(count-1)
            b[order[count:end]]=h[order[count:end]]*N/D
            break
        end
    end
   return b 
end

function optimalb!(h,M)
    totalh=sum(h)
    sumind=0
    order=sortperm(-h)
    
    if M*h[order[1]]./totalh <=1 
        copy!(b,M*h/totalh)
    end
    
    count=0
    sumh=0
    for ind in order
        count += 1
        sumh += h[ind]
        D=totalh - sumh
        N=M-count
        G=h[ind]*N/D
        if G <=1
            b[order[1:count-1]]=ones(count-1)
            b[order[count:end]]=h[order[count:end]]*N/D
            break
        end
    end

end

