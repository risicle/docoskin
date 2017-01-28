{
  pkgs ? import <nixpkgs> {},
  pythonPackages ? pkgs.python35Packages,
  forTest ? true
}:
{
  docoskinEnv = pkgs.stdenv.mkDerivation {
    name = "docoskin-env";
    buildInputs = pkgs.stdenv.lib.optional (!pythonPackages.isPy3k) pythonPackages.futures ++ [
      pythonPackages.opencv3
      pythonPackages.six
    ] ++ pkgs.stdenv.lib.optionals forTest [
      pythonPackages.pytest
      pythonPackages.pytestrunner
    ];
  };
}
