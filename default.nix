let
  # Look here for information about how to generate `nixpkgs-version.json`.
  #  → https://nixos.wiki/wiki/FAQ/Pinning_Nixpkgs
  pinnedVersion = pin: builtins.fromJSON (builtins.readFile pin);
  pinnedPkgs = pin:  import (builtins.fetchTarball {
    inherit (pinnedVersion pin) url sha256;
  }) {};
  pkgs' = pinned: (
    if (!isNull pinned) then pinnedPkgs pinned 
    else import <nixpkgs> {});

  hies-pkgs = import (builtins.fetchTarball {
    url = "https://github.com/domenkozar/hie-nix/tarball/master";
  });
in
{ pkgs ? pkgs' pinned, pinned ? null, enable-hie ? false }:
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

  libNN = python3Packages.buildPythonPackage rec {
    pname = "libNearestNeighbor";
    version = "1.0.0";
    propagatedBuildInputs = [ python3Packages.numpy ];
    src = ./nearestNeighbor;
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
      ptvsd
      numpy
      pandas
      scikitlearn
      matplotlib
      ipython
      pip
      virtualenv
      pylint
      autopep8
      pyqt5
      pycairo
      jupyter
      setuptools
  ]);
  # --------------- Commands ----------------


in {
  #inherit (pkgs)  libyaml libiconv;
  inherit pkgs;
  python-env = python-env;
} 
