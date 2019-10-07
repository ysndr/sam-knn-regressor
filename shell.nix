# This allows pinning by passing `--arg pin <pinfile.nix>.`
{ pin ? null } @ args: (import ./default.nix args).shell
