using Symbolics

function get_model_params(Je=10, delta=2, vthr=-50, Cm=200, vrest=-65, TauW=500, w0=0, a=6)	# model params
	mparams = SLVector((
		Je=10,
		delta = 2,
		vthr=-50,
		Cm=200,
		vrest=-65,
		TauW=500,
		w0=0,
		a=6
	))
	uparams = SLVector((
		v=mparams.vrest,
		w=mparams.w0
	))
	(mparams, uparams)
end

function instantiate_params_as_symbols(params)
    # Create expression for @parameters macro
    expr = Expr(:macrocall, Symbol("@parameters"))
    push!(expr.args, nothing)  # Required for macro calls

    # Add each parameter with its default value
    for name in LabelledArrays.symbols(mparams)
        push!(expr.args, :($name))
    end

    # Remember to eval expr on main scope
    expr
end
