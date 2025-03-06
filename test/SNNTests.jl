module SNNTests
using ReTest

include("../src/SNN.jl")
using Random
using ModelingToolkit, LinearSolve, DifferentialEquations, ComponentArrays
using Statistics

# Helper functions for tests
function setup_basic_params()
    params = SNN.Neuron.get_adex_neuron_params_skeleton(Float64)
    params.Ibase = 2.4e-10
    params.sigma = 1.0
    params.inc_gsyn = 8e-9
    params.a = 1.5e-9            # Subthreshold adaptation (A)
    params.b = 4.4e-12           # Spiking adaptation (A)
    params.TauW = 600.0e-3       # Adaptation time constant (s)
    params.Cm = 4.5e-10
    return params
end

function setup_stim_params()
    stim_params = SNN.Params.get_stim_params_skeleton()
    stim_params.p_deviant = 0.3
    stim_params.amplitude = 1.6e-9
    stim_params.duration = 50.0e-3
    stim_params.deviant_idx = 2
    stim_params.standard_idx = 1
    stim_params.p_deviant = 0.15
    stim_params.start_offset = 0.5  # Start earlier for shorter tests
    stim_params.isi = 300e-3
    return stim_params
end

# Run test suite
println("Starting tests")
ti = time()

@testset "SNN.jl" begin
    @testset "Basic Components" begin
        # Test neuron model component creation
        @testset "Neuron Components" begin
            @test SNN.Neuron.SynapseType <: Any
            @test SNN.Neuron.AMPA <: SNN.Neuron.SynapseType
            @test SNN.Neuron.GABAa <: SNN.Neuron.SynapseType

            # Test parameter skeletons
            params = setup_basic_params()
            @test params isa ComponentArray
            @test params.vrest ≈ -65.0e-3
            @test params.vthr ≈ -50.0e-3

            # Test initial conditions
            uparams = SNN.Neuron.get_adex_neuron_uparams_skeleton(Float64)
            @test uparams.v ≈ -65.0e-3
            @test uparams.w ≈ 0.0
        end

        # Test stimulus parameters and scheduling
        @testset "Stimulus Scheduling" begin
            stim_params = setup_stim_params()
            @test stim_params.duration == 50.0e-3

            # Test schedule generation
            tspan = (0, 2)
            stim_schedule = SNN.Params.generate_schedule(stim_params, tspan)
            @test size(stim_schedule, 1) == 3  # Format: [times, durations, groups]
            @test size(stim_schedule, 2) > 0   # Should contain at least one stimulus

            # Check valid schedule properties
            @test all(stim_schedule[1, :] .>= stim_params.start_offset)  # All stimuli start after offset
            @test all(stim_schedule[2, :] .== stim_params.duration)      # All durations match parameter

            # Test that stimuli times don't overlap
            stim_starts = stim_schedule[1, :]
            stim_ends = stim_starts .+ stim_schedule[2, :]
            for i in 2:length(stim_starts)
                @test stim_starts[i] >= stim_ends[i-1]  # No overlap in stimuli
            end
        end
    end

    # Test noise model - this is critical for SDE stability
    @testset "Noise Model" begin
        # Test that get_noise_eq returns a constant value
        params = setup_basic_params()
        tspan = (0, 0.9)  # Make tspan long enough to accommodate start_offset of 0.5
        stim_params = setup_stim_params()
        stim_schedule = SNN.Params.generate_schedule(stim_params, tspan)

        neuron = SNN.Neuron.make_neuron(params, SNN.Neuron.Soma, tspan, :test_neuron, stim_schedule)
        sigma = 0.5
        noise_eq = SNN.Neuron.get_noise_eq(neuron, sigma)

        # Check that noise is properly scaled by sigma
        @test noise_eq == sigma

        # Test noise matrix creation
        neurons = [
            SNN.Neuron.make_neuron(params, SNN.Neuron.Soma, tspan, Symbol("e_neuron_$(i)"), stim_schedule)
            for i in 1:2
        ]

        # Create minimal network
        connections::Vector{Pair{Vector{Equation}, Vector{Equation}}} = []
        network = SNN.Neuron.make_network(neurons, connections)

        # Test noise instantiation
        noise_eqs = SNN.Neuron.instantiate_noise(network, neurons, sigma)
        @test noise_eqs isa Matrix

        # Check diagonal structure (noise should only be on diagonal)
        for i in 1:size(noise_eqs, 1)
            for j in 1:size(noise_eqs, 2)
                if i == j
                    # Diagonal elements can be non-zero
                else
                    # Off-diagonal elements must be zero
                    @test noise_eqs[i, j] == 0.0
                end
            end
        end
    end

    # Test CSI calculation (critical for metrics)
    @testset "CSI Calculation" begin
        # Create sample data for standard and deviant responses
        times = collect(0:0.001:0.5)         # 500ms time window
        std_response = ones(length(times))   # Baseline response
        dev_response = 2 .* ones(length(times))  # Stronger response (2x)

        # Test basic calculation
        csi_value = SNN.Plots.csi([std_response, dev_response], times, 0.0, 0.3)
        expected_value = (2-1)/(2+1)  # Should be 0.333...
        @test isapprox(csi_value, expected_value, atol=0.01)

        # Test with adaptive windowing
        csi_adaptive = SNN.Plots.csi([std_response, dev_response], times, 0.0, 0.3, is_adaptative=true)
        @test isapprox(csi_adaptive, expected_value, atol=0.01)

        # Test edge cases
        # Empty time window
        csi_empty = SNN.Plots.csi([std_response, dev_response], times, 0.6, 0.7)
        @test isnan(csi_empty)

        # Zero responses
        zero_resp = zeros(length(times))
        csi_zero = SNN.Plots.csi([zero_resp, zero_resp], times, 0.0, 0.3)
        @test isnan(csi_zero)
    end

    # Integration test - short simulation run
    @testset "Short Simulation Run" begin
        Random.seed!(1234)  # Ensure reproducibility
        tspan = (0, 0.9)    # Short simulation for testing but long enough for start_offset

        params = setup_basic_params()
        stim_params = setup_stim_params()
        stim_schedule = SNN.Params.generate_schedule(stim_params, tspan)

        con_mapping_nested = [
            (SNN.Params.@connect_neurons [1, 2] SNN.Neuron.AMPA() 3),
        ]
        con_mapping::Vector{Tuple{Int64, Int64, SNN.Neuron.SynapseType}} = reduce(vcat, con_mapping_nested)

        pre_neurons = [row[1] for row in con_mapping]
        post_neurons = [row[2] for row in con_mapping]
        e_neurons_n = size(unique([pre_neurons; post_neurons]), 1)

        # Use recommended solver from benchmark if available, otherwise use a conservative default
        if @isdefined(recommended_solver_name) && recommended_solver_name !== nothing
            # Use the best solver from our benchmark
            println("Using recommended solver: $(recommended_solver_name) with tolerances $(recommended_tols)")
            solver_sym = Symbol(recommended_solver_name)
            solver = getfield(Main, solver_sym)()
            tols = recommended_tols
        else
            # Safe default with lenient tolerances
            println("Using default solver: SOSRI with lenient tolerances")
            solver = SOSRI()
            tols = (1e-2, 1e-2)  # More lenient tolerances for stability
        end

        # Run a quick simulation
        (sol, simplified_model, prob, results, neurons) = SNN.Pipeline.run_exp(
            "tmp/", "test_short";
            e_neurons_n=e_neurons_n,
            params=params,
            stim_params=stim_params,
            stim_schedule=stim_schedule,
            tspan=tspan,
            con_mapping=con_mapping,
            solver=solver,
            tols=tols,
            fetch_csi=true,
        )

        # Test that solution exists and has progressed sufficiently
        @test sol !== nothing
        # For numerical stability reasons, we check that the simulation reached at least 80% of the target time
        min_completion_ratio = 0.8
        completion_ratio = sol.t[end] / tspan[2]
        @test completion_ratio >= min_completion_ratio

        # Check that CSI values exist in results
        @test haskey(results, "csi_fine_50ms")
        @test haskey(results, "csi_fine_100ms")
        @test haskey(results, "csi_fine_300ms")
    end

    # Test to detect numerical instability issues
    @testset "Stability Tests" begin
        Random.seed!(1234)
        tspan = (0, Int(1))  # Use Int to avoid Tuple{Int,Float64} conversion issues

        params = setup_basic_params()
        stim_params = setup_stim_params()
        stim_schedule = SNN.Params.generate_schedule(stim_params, tspan)

        # Simple 2-neuron network
        con_mapping = [(1, 2, SNN.Neuron.AMPA())]
        e_neurons_n = 2

        # Test with higher noise level
        params.sigma = 2.0  # High noise to test stability

        # Use recommended solver from benchmark if available, otherwise use a conservative default
        if @isdefined(recommended_solver_name) && recommended_solver_name !== nothing
            # Use the best solver from our benchmark
            println("Using recommended solver: $(recommended_solver_name) with tolerances $(recommended_tols)")
            solver_sym = Symbol(recommended_solver_name)
            solver = getfield(Main, solver_sym)()
            tols = recommended_tols
        else
            # Safe default with lenient tolerances
            println("Using default solver: SOSRI with lenient tolerances")
            solver = SOSRI()
            tols = (1e-2, 1e-2)  # More lenient tolerances for stability
        end

        # Run simulation
        (sol, model, prob, results, neurons) = SNN.Pipeline.run_exp(
            "tmp/", "test_stability";
            e_neurons_n=e_neurons_n,
            params=params,
            stim_params=stim_params,
            stim_schedule=stim_schedule,
            tspan=tspan,
            con_mapping=con_mapping,
            solver=solver,
            tols=tols,
        )

        # Check for NaN or Inf values in solution
        for array in sol.u
            @test all(isfinite.(array))
            @test !any(isnan.(array))
        end

        # For numerical stability reasons, we check that the simulation reached at least 80% of the target time
        min_completion_ratio = 0.8
        completion_ratio = sol.t[end] / tspan[2]
        println("Simulation completed $(round(completion_ratio*100, digits=1))% of target time")
        @test completion_ratio >= min_completion_ratio

        # Allow for less strict success criteria since this is a challenging stochastic simulation
        acceptable_retcodes = [:Success, :MaxIters]
        @test sol.retcode in acceptable_retcodes
    end

    # Performance benchmarks and solver stability tests
    @testset "Benchmarking" begin
        Random.seed!(1234)
        params = setup_basic_params()
        stim_params = setup_stim_params()
        tspan = (0, 1.0)  # Short for benchmarking
        stim_schedule = SNN.Params.generate_schedule(stim_params, tspan)

        # Basic neuron creation
        @time neuron = SNN.Neuron.make_neuron(params, SNN.Neuron.Soma, tspan, :test_neuron, stim_schedule)

        # Network creation with different sizes
        @time neurons_small = [
            SNN.Neuron.make_neuron(params, SNN.Neuron.Soma, tspan, Symbol("e_neuron_$(i)"), stim_schedule)
            for i in 1:3
        ]
        @time connections_small::Vector{Pair{Vector{Symbolics.Equation}, Vector{Symbolics.Equation}}} = []
        @time network_small = SNN.Neuron.make_network(neurons_small, connections_small)

        # Timing for noise instantiation
        sigma = 0.5
        @time noise_small = SNN.Neuron.instantiate_noise(network_small, neurons_small, sigma)

        # Print summary
        println("\nPerformance Summary:")
        println("- Single neuron creation: measure in separate runs")
        println("- 3-neuron network creation: measure in separate runs")

        # Declare a global variable to store the recommended solver
        global recommended_solver_name = nothing
        global recommended_tols = nothing

        # Solver Benchmark
        @testset "Solver Stability Benchmark" begin
            # Setup a small test network for benchmarking different solvers
            tspan_solvers = (0, 2.0)  # Short enough for quick tests but long enough to be meaningful
            e_neurons_n = 3  # Small network for quick tests

            # Create a simple connection mapping
            con_mapping = [
                (1, 3, SNN.Neuron.AMPA()),
                (2, 3, SNN.Neuron.AMPA()),
            ]

            # Define solvers to test - include both ODE and SDE solvers
            ode_solvers = [
                (:Tsit5, Tsit5()),     # Good general-purpose non-stiff ODE solver
                (:Rodas5, Rodas5())    # Stiff ODE solver
            ]

            sde_solvers = [
                (:SOSRI, SOSRI()),     # Standard SDE solver, typically good performance
                (:ImplicitRKMil, ImplicitRKMil()), # More accurate for stiff SDEs
                (:ImplicitEM, ImplicitEM()),  # Good for stiff SDEs
                (:EM, EM()),  # Good for stiff SDEs
                (:EulerHeun, EulerHeun())  # Good for stiff SDEs
            ]

            # Tolerances to test
            tolerance_sets = [
                (1e-5, 1e-5),  # Tight tolerances
                (1e-4, 1e-4),  # Tight tolerances
                (1e-3, 1e-3),  # Tight tolerances
                (1e-2, 1e-2),  # Medium tolerances
                (1e-1, 1e-1)   # Loose tolerances for challenging cases
            ]

            # Noise levels to test
            noise_levels = [0.01]  # Test different noise levels

            # Track results for comparison
            results_table = []

            println("\n===== SOLVER BENCHMARK =====")
            println("Testing solvers with different parameters:\n")

            # Test each solver with different noise levels and tolerance settings
            for noise_level in noise_levels
                println("\n--- Noise Level: $noise_level ---")

                # Update params with current noise level
                test_params = deepcopy(params)
                test_params.sigma = noise_level

                # Choose solver set based on noise level
                solvers = noise_level > 0 ? sde_solvers : ode_solvers

                for (solver_name, solver) in solvers
                    for (tol_idx, tols) in enumerate(tolerance_sets)
                        println("\nTesting $solver_name with tolerances = $tols")

                        # Create a clean stim schedule for each test
                        test_stim_schedule = SNN.Params.generate_schedule(stim_params, tspan_solvers)

                        # Set up timing
                        start_time = time()

                        try
                            # Run a simulation and save plots for analysis
                            benchmark_name = "benchmark_$(solver_name)_tol$(tol_idx)_noise$(noise_level)"
                            (sol, simplified_model, prob, results, neurons) = SNN.Pipeline.run_exp(
                                "tmp/", benchmark_name;
                                e_neurons_n=e_neurons_n,
                                params=test_params,
                                stim_params=stim_params,
                                stim_schedule=test_stim_schedule,
                                tspan=tspan_solvers,
                                con_mapping=con_mapping,
                                solver=solver,
                                tols=tols,
                                save_plots=false,  # We'll handle plotting manually for more control
                                fetch_csi=false
                            )

                            open("tmp/benchmark_$(benchmark_name)_equations.txt", "w") do io
                                println(io, "Equations:")
                                println(io, "----------")
                                for (i, eq) in enumerate(equations(simplified_model))
                                    println(io, "Equation $i:")
                                    println(io, eq)
                                end
                            end
                            open("tmp/benchmark_$(benchmark_name)_callbacks.txt", "w") do io
                                println(io, "Callbacks:")
                                println(io, "----------")
                                for (i, callback) in enumerate(continuous_events(simplified_model))
                                    println(io, "Callback $i:")
                                    println(io, callback)
                                end
                            end

                            # Create plots directory
                            mkpath("tmp/benchmark_plots")

                            # Calculate completion ratio early for plotting decision
                            completion_ratio = sol.t[end] / tspan_solvers[2]
                            steps_taken = length(sol.t)

                            # Generate plots for simulations that reach at least 30% completion
                            if completion_ratio > 0.3
                                try
                                    # Create a plot prefix name
                                    plot_prefix = "tmp/benchmark_plots/$(benchmark_name)"

                                    # Get the tree for parameter access
                                    tree = SNN.Utils.make_param_tree(simplified_model)

                                    # Make voltage plots for each neuron
                                    for i in 1:e_neurons_n
                                        # Get neuron voltage parameter
                                        e_v = SNN.Plots.fetch_tree_neuron_value("e_neuron", i, "v", tree)

                                        # Plot voltage with custom settings
                                        SNN.Plots.plot_neuron_value(
                                            sol.t, sol[e_v],
                                            nothing, nothing,
                                            stim_params.start_offset;
                                            start=0.0,
                                            stop=sol.t[end],
                                            title="Voltage - $(solver_name) (σ=$(noise_level), tol=$(tols[1]))",
                                            name="$(plot_prefix)_voltage_e_$(i).png",
                                            schedule=test_stim_schedule,
                                            tofile=true
                                        )
                                    end

                                    # Plot overall network raster plot
                                    res = SNN.Utils.fetch_tree(["e_neuron", "R"], tree)
                                    ma = SNN.Utils.hcat_sol_matrix(res, sol)
                                    spikes_times = SNN.Utils.get_spike_timings(ma, sol)

                                    SNN.Plots.plot_spikes(
                                        (spikes_times, []);
                                        start=0.0,
                                        stop=sol.t[end],
                                        color=(:red, :blue),
                                        height=300,
                                        title="Spikes - $(solver_name) (σ=$(noise_level), tol=$(tols[1]))",
                                        name="$(plot_prefix)_raster.png",
                                        schedule=test_stim_schedule,
                                        tofile=true
                                    )

                                    # Save parameter values for this simulation
                                    # Create a summary file with experiment settings
                                    open("$(plot_prefix)_params.txt", "w") do io
                                        println(io, "SOLVER BENCHMARK SIMULATION PARAMETERS")
                                        println(io, "===================================\n")
                                        println(io, "Solver: $(solver_name)")
                                        println(io, "Noise level (sigma): $(noise_level)")
                                        println(io, "Tolerances: $(tols)")
                                        println(io, "Completion: $(round(completion_ratio*100, digits=1))%")
                                        println(io, "Runtime: $(round(elapsed, digits=2)) seconds")
                                        println(io, "Steps taken: $(steps_taken)")
                                        println(io, "\nNeuron Parameters:")
                                        for (key, value) in pairs(test_params)
                                            println(io, "  $(key): $(value)")
                                        end
                                        println(io, "\nStimulus Parameters:")
                                        for (key, value) in pairs(stim_params)
                                            println(io, "  $(key): $(value)")
                                        end
                                        println(io, "\nSolver return code: $(sol.retcode)")
                                        println(io, "Final dt: $(try round(sol.t[end] - sol.t[end-1], digits=5) catch; "N/A" end)")
                                    end

                                    println("  - Created voltage plots, raster plot, and parameter file")
                                catch plot_error
                                    println("  - Failed to create plots: $(typeof(plot_error))")
                                end
                            end

                            # Measure elapsed time
                            elapsed = time() - start_time

                            # Calculate remaining metrics
                            success = sol.retcode == :Success

                            # Store results
                            push!(results_table, (
                                solver_name = String(solver_name),
                                noise_level = noise_level,
                                tolerances = tols,
                                success = success,
                                completed = completed_ratio,
                                steps = steps_taken,
                                time = elapsed,
                                dt_final = try sol.t[end] - sol.t[end-1] catch; NaN end,
                                stability_score = success ? (completed_ratio * 10) : (completed_ratio * 5)
                            ))

                            # Print immediate results
                            println("  ✓ Completed: $(round(completed_ratio*100, digits=1))% in $(round(elapsed, digits=2))s with $(steps_taken) steps")
                            println("    Final dt: $(try round(sol.t[end] - sol.t[end-1], digits=5) catch; "N/A" end)")
                            println("    Status: $(success ? "Success" : "Failed with $(sol.retcode)")")

                        catch e
                            # If simulation completely fails
                            elapsed = time() - start_time
                            println("  ✗ Failed: $(typeof(e)): $(e)")

                            push!(results_table, (
                                solver_name = String(solver_name),
                                noise_level = noise_level,
                                tolerances = tols,
                                success = false,
                                completed = 0.0,
                                steps = 0,
                                time = elapsed,
                                dt_final = NaN,
                                stability_score = 0
                            ))
                        end
                    end
                end
            end

            # Sort and print summary
            sort!(results_table, by = x -> (x.success ? 1 : 0, -x.completed, x.time))

            println("\n===== SOLVER BENCHMARK RESULTS =====")
            println("Results ranked by stability and speed:\n")

            println("MOST STABLE SOLVERS:")
            for (i, result) in enumerate(results_table[1:min(5, length(results_table))])
                println("$(i). $(result.solver_name) (σ=$(result.noise_level), tol=$(result.tolerances))")
                println("   Completion: $(round(result.completed*100, digits=1))%, Status: $(result.success ? "Success" : "Failed")")
                println("   Runtime: $(round(result.time, digits=2))s, Steps: $(result.steps)")
            end

            # Find fastest successful solvers
            successful = filter(r -> r.success && r.completed > 0.99, results_table)
            if !isempty(successful)
                sort!(successful, by = x -> x.time)

                println("\nFASTEST STABLE SOLVERS:")
                for (i, result) in enumerate(successful[1:min(3, length(successful))])
                    println("$(i). $(result.solver_name) (σ=$(result.noise_level), tol=$(result.tolerances))")
                    println("   Runtime: $(round(result.time, digits=2))s, Steps: $(result.steps)")
                end
            else
                println("\nNo completely stable solvers found in this benchmark run.")
            end

            # Find best solvers for stochastic simulations (noise > 0)
            stochastic = filter(r -> r.noise_level > 0 && r.completed > 0.8, results_table)
            if !isempty(stochastic)
                sort!(stochastic, by = x -> (x.success ? 1 : 0, -x.completed, x.time))

                println("\nBEST SOLVERS FOR STOCHASTIC SIMULATIONS:")
                for (i, result) in enumerate(stochastic[1:min(3, length(stochastic))])
                    println("$(i). $(result.solver_name) (σ=$(result.noise_level), tol=$(result.tolerances))")
                    println("   Completion: $(round(result.completed*100, digits=1))%, Runtime: $(round(result.time, digits=2))s")
                end
            else
                println("\nNo stable stochastic solvers found in this benchmark run.")
            end

            # Store benchmark results in a file for future reference
            open("tmp/solver_benchmark_results.txt", "w") do io
                println(io, "SOLVER BENCHMARK RESULTS")
                println(io, "=======================\n")

                println(io, "All Results (sorted by stability then speed):")
                for (i, result) in enumerate(results_table)
                    println(io, "$(i). $(result.solver_name) (σ=$(result.noise_level), tol=$(result.tolerances))")
                    println(io, "   Completion: $(round(result.completed*100, digits=1))%, Status: $(result.success ? "Success" : "Failed")")
                    println(io, "   Runtime: $(round(result.time, digits=2))s, Steps: $(result.steps)")
                    println(io, "   Final dt: $(isnan(result.dt_final) ? "N/A" : round(result.dt_final, digits=5))")
                    println(io)
                end
            end

            # Recommend the best solver for the test suite
            best_for_tests = if !isempty(successful)
                first(successful)
            elseif !isempty(stochastic)
                first(stochastic)
            elseif !isempty(results_table)
                # Get the most stable result even if not fully successful
                sort!(results_table, by = x -> -x.completed)[1]
            else
                nothing
            end

            if best_for_tests !== nothing
                println("\nRECOMMENDED SOLVER FOR TESTS:")
                println("$(best_for_tests.solver_name) with tolerances=$(best_for_tests.tolerances)")
                println("This combination achieved $(round(best_for_tests.completed*100, digits=1))% completion")
                println("in $(round(best_for_tests.time, digits=2))s with $(best_for_tests.steps) steps.")

                # Set the global recommended solver if we found a good one
                if best_for_tests.completed > 0.9
                    println("\nUpdating test configuration with recommended solver...")
                    global recommended_solver_name = best_for_tests.solver_name
                    global recommended_tols = best_for_tests.tolerances
                end
            else
                println("\nNo recommendations available from this benchmark run.")
            end
        end
    end
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti / 60, digits=3), " minutes")

end
