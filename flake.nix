{
  description = "Shell julia cuda";

  # Flake inputs
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  # Flake outputs
  outputs = { self, nixpkgs, flake-utils }:
    let
      # Systems supported
      allSystems = [
        "x86_64-linux" # 64-bit Intel/AMD Linux
      ];

      # Helper to provide system-specific attributes
      forAllSystems = f: nixpkgs.lib.genAttrs allSystems (system: f {
        pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
        system = system;
      });
    in
    {
      # Development environment output
      devShells = forAllSystems ({ pkgs, system }: {
        default = pkgs.mkShell {
          # The Nix packages provided in the environment
          packages = with pkgs; [
            # Python plus helper tools
            zlib
            ripgrep
            gcc11
            just
            julia
            uncrustify
          ];
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.zlib}/lib:${pkgs.stdenv.cc.cc.lib}/lib
            export EXTRA_CCFLAGS="-I/usr/include"

            unset SOURCE_DATE_EPOCH
          '';
          };
      });
    };
}
