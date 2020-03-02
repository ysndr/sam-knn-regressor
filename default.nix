{pkgs ? import (if pin == false then <nixpkgs> else pin) {},
 pin ? ./nixpkgs.nix,
 vscode-debug ? false, ... }:
with pkgs;
with  import pkgs.path {
  overlays = [
    (self: super: {
      enchant = super.enchant.overrideAttrs (oldAttrs: rec {
        postConfigure = ''
        substituteInPlace src/Makefile --replace \
         '$(AM_V_lt) $(AM_LIBTOOLFLAGS)'\
         '$(AM_V_lt) --tag=CC $(AM_LIBTOOLFLAGS)'
        '';
      });
    })];
  };
let
  # ------------- Python ----------------
  # for build usage only

  python' = python3.override {
    packageOverrides = self: super: {
      pylint = super.pylint.overridePythonAttrs(old: rec {
        doCheck = false;
      });
      scikitlearn = super.scikitlearn.overridePythonAttrs( old: rec {
        doCheck = false;
        checkPhase = ":";
      });
    };
  };

  scikit-multiflow =  python'.pkgs.buildPythonPackage rec {
    pname = "scikit-multiflow";
    version = "0.3.0";
    doCheck = false;
    propagatedBuildInputs = with python'.pkgs; [ numpy pandas matplotlib sortedcontainers scikitlearn];

    src = python3Packages.fetchPypi {
      inherit pname version;
      sha256 = "0cba75131vwsfzrb56i6qvxi0j6ma7rqq4y6896s331sxkrg1zbh";
    };
  };

  ptvsd = python3Packages.buildPythonPackage rec {
    pname = "ptvsd";
    version = "4.3.2";
    doCheck = false;
    propagatedBuildInputs = with python3Packages; [ ];
    format = "wheel";
    src = python3Packages.fetchPypi {
      inherit pname version;
      sha256 = "13g685gjbfpwvzahz3m3d46v770n70kyvcj0a18h5fv8c1rkg4a5";
      format = "wheel";
    };
  };

  python-env = python'.withPackages(pp: with pp; [
      scikit-multiflow
      numpy
      pandas
      scikitlearn
      matplotlib
      ipython
      pip
      virtualenv
      pylint
      autopep8
      tkinter
      pycairo
      jupyter
      setuptools
      pykdtree
    ] ++ lib.optional vscode-debug ptvsd
  );
  # --------------- Commands ----------------
  shell = mkShell {
    name = "SAMkNNReg-env";
    buildInputs = [ python-env ];

    shellHook = ''
      export MPLBACKEND="tkAgg"
    '';
  };

in {
  inherit shell;
}
