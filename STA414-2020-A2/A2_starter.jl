using Revise # lets you change A2funcs without restarting julia!
includet("/Users/mac/Downloads/STA414-2020-A2-skyqqq-master/A2_src.jl")
using Plots
using Statistics: mean
using Zygote
using Test
using Logging
using .A2funcs: log1pexp # log(1 + exp(x)) stable
using .A2funcs: factorized_gaussian_log_density
using .A2funcs: skillcontour!
using .A2funcs: plot_line_equal_skill!
using Distributions

function log_prior(zs)
  return factorized_gaussian_log_density(0,0,zs)
end

function logp_a_beats_b(za,zb)
  return -log1pexp.(-(za-zb))
end


function all_games_log_likelihood(zs,games)
  zs_a = zs[games[:,1],:]
  zs_b = zs[games[:,2],:]
  likelihoods = logp_a_beats_b.(zs_a,zs_b)
  return  sum(likelihoods, dims=1)
end

function joint_log_density(zs,games)
  return sum(all_games_log_likelihood(zs,games).+log_prior(zs),dims=1)
end

@testset "Test shapes of batches for likelihoods" begin
  B = 15 # number of elements in batch
  N = 4 # Total Number of Players
  test_zs = randn(4,15)
  test_games = [1 2; 3 1; 4 2] # 1 beat 2, 3 beat 1, 4 beat 2
  @test size(test_zs) == (N,B)
  #batch of priors
  @test size(log_prior(test_zs)) == (1,B)
  # loglikelihood of p1 beat p2 for first sample in batch
  @test size(logp_a_beats_b(test_zs[1,1],test_zs[2,1])) == ()
  # loglikelihood of p1 beat p2 broadcasted over whole batch
  @test size(logp_a_beats_b.(test_zs[1,:],test_zs[2,:])) == (B,)
  # batch loglikelihood for evidence
  @test size(all_games_log_likelihood(test_zs,test_games)) == (1,B)
  # batch loglikelihood under joint of evidence and prior
  @test size(joint_log_density(test_zs,test_games)) == (1,B)
end

# Convenience function for producing toy games between two players.
two_player_toy_games(p1_wins, p2_wins) = vcat([repeat([1,2]',p1_wins), repeat([2,1]',p2_wins)]...)

# Example for how to use contour plotting code
#plot(title="Example Gaussian Contour Plot",
#    xlabel = "Player 1 Skill",
#    ylabel = "Player 2 Skill"
#   )
#example_gaussian(zs) = exp(factorized_gaussian_log_density([-1.,2.],[0.,0.5],zs))
#skillcontour!(example_gaussian)
#plot_line_equal_skill!()
#savefig(joinpath("plots","example_gaussian.pdf"))


zs = randn(2,15)
# plot prior contours
plot(title="Prior Contour Plot",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill"
   )
prior(zs) = exp(log_prior(zs))
skillcontour!(zs -> prior(zs))
plot_line_equal_skill!()
savefig(joinpath("plots","joint_prior.pdf"))

# plot likelihood contours
game12 = two_player_toy_games(1,2)
plot(title="Likelihood Contour Plot",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill"
   )
likelihood(zs) = exp(all_games_log_likelihood(zs,game12))
skillcontour!(zs -> likelihood(zs))
plot_line_equal_skill!()
savefig(joinpath("plots","likelihood.pdf"))

# plot joint contours with player A winning 1 game
games = two_player_toy_games(1,0)
plot(title="Joint Posterior1_0 Contour Plot",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill"
   )
joint1(zs) = exp(joint_log_density(zs,games))
skillcontour!(zs -> joint1(zs))
plot_line_equal_skill!()
savefig(joinpath("plots","joint1.pdf"))
# plot joint contours with player A winning 10 games
games = two_player_toy_games(10,0)
plot(title="Joint Posterior10_0 Contour Plot",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill"
   )
joint10(zs) = exp(joint_log_density(zs,games))
skillcontour!(zs -> joint10(zs))
plot_line_equal_skill!()
savefig(joinpath("plots","joint10.pdf"))

# plot joint contours with player A winning 10 games and player B winning 10 games
games = two_player_toy_games(10,10)
plot(title="Joint Posterior10_10 Contour Plot",
    xlabel = "Player A Skill",
    ylabel = "Player B Skill"
   )
joint20(zs) = exp(joint_log_density(zs,games))
skillcontour!(zs -> joint20(zs))
plot_line_equal_skill!()
savefig(joinpath("plots","joint20.pdf"))

function elbo(params,logp,num_samples)
  samples = exp.(params[2]) .*randn(length(params[1]),num_samples) .+params[1]
  logp_estimate = logp(samples)
  logq_estimate = factorized_gaussian_log_density(params[1],params[2],samples)
  return sum(logp_estimate-logq_estimate)/num_samples
end

# Conveinence function for taking gradients
function neg_toy_elbo(params; games = two_player_toy_games(1,0), num_samples = 100)
  logp(zs) = joint_log_density(zs,games)
  return -elbo(params,logp, num_samples)
end


# Toy game
num_players_toy = 2
toy_mu = [-2.,3.] # Initial mu, can initialize randomly!
toy_ls = [0.5,0.] # Initual log_sigma, can initialize randomly!
toy_params_init = (toy_mu, toy_ls)


function fit_toy_variational_dist(init_params, toy_evidence; num_itrs=200, lr= 1e-2, num_q_samples = 10)
  params_cur = init_params
  for i in 1:num_itrs
    grad_params = gradient(params -> neg_toy_elbo(params;games = toy_evidence, num_samples = num_q_samples), params_cur)[1]#gradients of variational objective with respect to parameters
    params_cur =  params_cur .- lr.* grad_params #update paramters with lr-sized step in descending gradient
    @info "nelbo: $(neg_toy_elbo(params_cur; games=toy_evidence, num_samples=num_q_samples))"#report the current elbbo during training
    # plot true posterior in red and variational in blue
    plot();
    #plot likelihood contours for target posterior
    display(skillcontour!(zs->exp(joint_log_density(zs, toy_evidence)),colour=:red))
    plot_line_equal_skill!()
    #plot likelihood contours for variational posterior
    display(skillcontour!(zs -> exp(factorized_gaussian_log_density(params_cur[1], params_cur[2],zs)),colour=:blue))
  end
  return params_cur
end

#fit q with SVI observing player A winning 1 game
game10 = two_player_toy_games(1,0)
fit_toy_variational_dist(toy_params_init,game10)
xlabel!("Player A skill")
ylabel!("Player B skill")
title!("target posterior (red) and variational posterior (blue)")
savefig(joinpath("plots","posterior 1.pdf"))

#fit q with SVI observing player A winning 10 games
game100 = two_player_toy_games(10,0)
fit_toy_variational_dist(toy_params_init,game100)
xlabel!("Player A skill")
ylabel!("Player B skill")
title!("target posterior (red) and variational posterior (blue)")
savefig(joinpath("plots","posterior 10.pdf"))

#fit q with SVI observing player A winning 10 games and player B winning 10 games
game1010 = two_player_toy_games(10,10)
fit_toy_variational_dist(toy_params_init,game1010)
xlabel!("Player A skill")
ylabel!("Player B skill")
title!("target posterior (red) and variational posterior (blue)")
savefig(joinpath("plots","posterior 20.pdf"))

## Question 4
# Load the Data
using MAT
vars = matread("/Users/mac/Downloads/STA414-2020-A2-skyqqq-master/tennis_data.mat")
player_names = vars["W"]
tennis_games = Int.(vars["G"])
num_players = length(player_names)
print("Loaded data for $num_players players")


function fit_variational_dist(init_params, tennis_games; num_itrs=200, lr= 1e-2, num_q_samples = 10)
  params_cur = init_params
  for i in 1:num_itrs
    grad_params = gradient(params -> neg_toy_elbo(params;games = tennis_games, num_samples = num_q_samples), params_cur)[1]#gradients of variational objective wrt params
    params_cur = params_cur .- lr.*grad_params #update parmaeters wite lr-sized steps in desending gradient direction
    @info "nelbo: $(neg_toy_elbo(params_cur; games=tennis_games, num_samples=num_q_samples))"#report objective value with current parameters
  end
  return params_cur
end

# TODO: Initialize variational family
init_mu = randn(107)#random initialziation
init_log_sigma = randn(107)# random initialziation
init_params = (init_mu, init_log_sigma)

# Train variational distribution
trained_params = fit_variational_dist(init_params, tennis_games)


#10 players with highest mean skill under variational model
#hint: use sortperm
means = trained_params[1]
logsd = trained_params[2]
perm = sortperm(means, rev=true)
plot(means[perm], yerror = exp.(logsd[perm]))
xlabel!("Sorted Player by skills")
ylabel!("players' skills")
title!("Approx. mean (line) and vairance (vertical range) sorted by skills")
savefig(joinpath("plots","mean_variance under variational model.pdf"))

# Find top 10 hight means
print(player_names[perm[1:10]])

#joint posterior over "Roger-Federer" and ""Rafael-Nadal""
#hint: findall function to find the index of these players in player_names
indexRF = findall(name-> name == "Roger-Federer", player_names)
indexRN = findall(name-> name == "Rafael-Nadal", player_names)
plot(legend=:bottomright)
skillcontour!(zs -> exp(factorized_gaussian_log_density([means[1], means[5]],[logsd[1],logsd[5]],zs)))
plot_line_equal_skill!()
xlabel!("Roger-Federer skill")
ylabel!("Rafael-Nadal skill")
title!("Joint Posterior Isocontour Plot")
savefig(joinpath("plots","Joint Posterior Isocontour Plot.pdf"))

# P(zRF>zRN)
print(1-cdf(Normal(means[5]-means[1], sqrt((exp(logsd[1]))^2+(exp(logsd[5]))^2)),0))
# P(zRF>zb)
print(1-cdf(Normal(means[5]-means[75], sqrt((exp(logsd[75]))^2+(exp(logsd[5]))^2)),0))

# P(zRF>zRN)
zs_RF = rand(Normal(means[5], exp(logsd[5])),10000)
zs_RN = rand(Normal(means[1], exp(logsd[1])),10000)
print(sum(zs_RF .> zs_RN,dims=1) / 10000)

# P(zRF>zb)
lowestindex = perm[107]
zs_RF = rand(Normal(means[5], exp(logsd[5])),10000)
zs_b = rand(Normal(means[75], exp(logsd[75])),10000)
print(sum(zs_RF .> zs_b,dims=1) / 10000)
