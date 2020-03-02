# This allows pinning by passing `--arg pin <pinfile.nix>.`
{ pin ? null, vscode-debug ? false } @ args: (import ./default.nix args).shell
