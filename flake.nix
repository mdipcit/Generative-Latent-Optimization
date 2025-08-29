{
  description = "Python venv development template";

  inputs = {
    utils.url = "github:numtide/flake-utils";
    BSDS500 = {
      url = "https://github.com/mdipcit/image_resize_pipeline/releases/download/v1.0-bsds500/BSDS500_512x512_full.tar.gz";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      utils,
      BSDS500,
      ...
    }:
    utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        packages = {
          bsds500 = pkgs.runCommand "bsds500-src" { src = BSDS500; } ''cp -r $src $out'';
        };
        devShells.default = pkgs.mkShell {
          name = "python-uv";
      
          buildInputs = [
            # A Python interpreter including the 'venv' module is required to bootstrap
            # the environment.
            pkgs.python310
            pkgs.uv


            # Those are dependencies that we would like to use from nixpkgs, which will
            # add them to PYTHONPATH and thus make them accessible from within the venv.
            pkgs.zlib
            pkgs.stdenv.cc.cc.lib
            pkgs.libGL
            pkgs.glib
            pkgs.wget
            pkgs.cudatoolkit
            pkgs.graphviz
          ];
          env = {
            LD_LIBRARY_PATH = "${
              with pkgs;
              lib.makeLibraryPath [
                zlib
                stdenv.cc.cc.lib
                libGL
                glib
              ]
            }:/run/opengl-driver/lib";
            BSDS500_PATH = "${self.packages.${system}.bsds500}";
          };


          ShellHook = ''
            export CUDA_PATH=${pkgs.cudatoolkit}
          '';
        };
      }
    );
}
