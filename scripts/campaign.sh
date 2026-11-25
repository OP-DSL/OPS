#!/usr/bin/env bash
# One entry point for the LUMI campaigns of CloverLeaf_3D, SENGA2 and TGV.
#
#   bash campaign.sh <cloverleaf|senga2|tgv|tgv-top3> <new|continue|retry|status|list> [options]
#
#   cloverleaf, senga2, tgv   MPI-decomposition sweep: 1440^3 on 20 nodes, default / autotune / precomputed
#                             runs for every decomposition (tgv is tgvstorenone, package lumi_tgv_decomp)
#   tgv-top3                  the earlier TGV launcher (lumi_tgv_top3): replay of the TOP3 dataset at 2160^3
#
#   new       create a campaign, build only if needed, freeze the plan and submit
#   continue  submit what is still missing in the latest campaign (or the one given with --campaign)
#   retry     like continue, and failed runs are run again
#   status    progress of the campaign and your jobs in the queue
#   list      campaigns of the application, newest first
#
# Options (all optional):
#   --campaign NAME   campaign to act on: full path, folder name or any part of it (e.g. 20261118_153312)
#   --steps N         time steps per run            (default: cloverleaf 2250, senga2 100, tgv 2800)
#   --grid N          cubic grid                    (default: 1440, the package default)
#   --nodes N         nodes; tgv-top3: a list 2,3,4 (default: 20; tgv-top3 2,3,4,6,8,10,12,16,20,24,32,40)
#   --jobs N          chained allocations           (default: 2)
#   --decompositions L  only these MPI decompositions, e.g. 8x5x4,10x2x8 (with new; default: the package set).
#                     Include 8x5x4 (the default of OPS) when the result must be compared against it
#   --time T          wall time of each allocation  (default: the package default; tgv-top3 01:00:00)
#   --account A       Slurm project; otherwise $SLURM_ACCOUNT, otherwise ACCOUNT= in ~/.ops_campaign.conf
#   --rebuild         compile even when an identical build of an earlier campaign could be reused
#   --no-build        never compile: reuse the executable of the newest earlier campaign with the same
#                     application and grids (cloverleaf: any grid, its executable reads the grid at run
#                     time), even if its OPS commit, environment or checkout differ.
#                     The differences are printed and saved in <campaign>/build_reused.json
#   --dry-run         print the commands without running them
#
# The executable of an earlier campaign is reused when the OPS commit, the environment file, the grids
# and the build code of the package are the same and that checkout had no uncommitted changes.
# tgv-top3 takes its cases from the dataset: --steps, --grid, --jobs, --decompositions and --rebuild do
# not apply to it. The tgv default of 2800 steps is an estimate for runs of about 5 minutes (500 steps
# at 2160^3 on 20 nodes took about 181 s; scaled by the number of grid points).
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin="${OPS_LAUNCHER_PYTHON:-python3}"

die() { echo "ERROR: $*" >&2; exit 2; }
usage() { sed -n '2,36p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

[ $# -ge 1 ] || usage 2
case "$1" in -h|--help) usage 0;; esac
[ $# -ge 2 ] || usage 2
app=$1; action=$2; shift 2
campaign_arg=''; steps=''; grid=''; nodes=''; jobs=2; walltime=''; account=''; rebuild=0; nobuild=0; dry=0; decomps=''
while [ $# -gt 0 ]; do
  case "$1" in
    --campaign|--steps|--grid|--nodes|--jobs|--time|--account|--decompositions)
      [ $# -ge 2 ] || die "$1 needs a value"
      case "$1" in
        --campaign) campaign_arg=$2;; --steps) steps=$2;; --grid) grid=$2;; --nodes) nodes=$2;;
        --jobs) jobs=$2;; --time) walltime=$2;; --account) account=$2;; --decompositions) decomps=$2;;
      esac
      shift 2;;
    --rebuild) rebuild=1; shift;;
    --no-build) nobuild=1; shift;;
    --dry-run) dry=1; shift;;
    -h|--help) usage 0;;
    *) die "Unknown option $1 (see: bash $0 --help)";;
  esac
done

case "$app" in
  cloverleaf) package=lumi_cloverleaf_decomp; folder=ops-cloverleaf-decomp; prefix=CLOVERLEAF_DECOMP_; default_steps=2250;;
  senga2)     package=lumi_senga2_decomp;     folder=ops-senga2-decomp;     prefix=SENGA2_DECOMP_;     default_steps=100;;
  tgv)        package=lumi_tgv_decomp;        folder=ops-tgv-decomp;        prefix=TGV_DECOMP_;        default_steps=2800;;
  tgv-top3)   package=lumi_tgv_top3;          folder=ops-decomposition;     prefix=TGV_TOP3_;          default_steps='';;
  *) die "Unknown application '$app': use cloverleaf, senga2, tgv or tgv-top3";;
esac
override='campaign.sh: qualified in earlier campaigns'
[ "$app" != tgv ] || override='campaign.sh: submitted without the qualification stage of this campaign'
[ "$rebuild" = 0 ] || [ "$nobuild" = 0 ] || die "--rebuild and --no-build exclude each other"
[ "$app" != tgv-top3 ] || [ "$nobuild" = 0 ] || die "--no-build does not apply to tgv-top3"
case "$action" in new|continue|retry|status|list) ;; *) die "Unknown action '$action': use new, continue, retry, status or list";; esac
[ -z "$steps" ] && steps=$default_steps
if [ -n "$decomps" ]; then
  [ "$app" != tgv-top3 ] || die "--decompositions does not apply to tgv-top3 (its cases come from the dataset)"
  [ "$action" = new ] || die "--decompositions only applies to 'new': the plan of an existing campaign is frozen. \
'continue' already runs only what is missing and 'retry' also repeats the failed runs."
fi

if [ -z "$account" ]; then account=${SLURM_ACCOUNT:-}; fi
if [ -z "$account" ] && [ -f "$HOME/.ops_campaign.conf" ]; then
  account=$(sed -n 's/^ACCOUNT=//p' "$HOME/.ops_campaign.conf" | head -1)
fi
[ -n "$account" ] || die "No Slurm project. Save it once with:  echo ACCOUNT=<your project> > ~/.ops_campaign.conf"
root=${OPS_CAMPAIGN_ROOT:-/scratch/$account/$folder/campaigns}

run() {                       # print the command; run it unless --dry-run
  printf '+'; printf ' %q' "$@"; printf '\n'
  [ "$dry" = 1 ] || "$@"
}

campaigns() {                 # campaign folders of this application, newest first (the name carries the date)
  [ -d "$root" ] || return 0
  find "$root" -mindepth 2 -maxdepth 2 -name campaign.json 2>/dev/null | xargs -r -n1 dirname | grep "/$prefix" | sort -r
}

pick_campaign() {             # the one given with --campaign, or the newest
  local found
  if [ -n "$campaign_arg" ]; then
    if [ -f "$campaign_arg/campaign.json" ]; then (cd "$campaign_arg" && pwd); return; fi
    found=$(campaigns | grep -F -- "$campaign_arg" || true)
    [ -n "$found" ] || die "No $app campaign matches '$campaign_arg' in $root (see: bash $0 $app list)"
    [ "$(printf '%s\n' "$found" | wc -l)" = 1 ] || die "'$campaign_arg' matches several campaigns:
$found"
    printf '%s\n' "$found"
  else
    found=$(campaigns | head -1)
    [ -n "$found" ] || die "No $app campaign in $root. Create one with: bash $0 $app new"
    printf '%s\n' "$found"
  fi
}

built() {                     # 0 when every build of the campaign has its manifest
  local c=$1 n
  case "$app" in
    cloverleaf|tgv) [ -f "$c/build/hip/build_manifest.json" ];;
    senga2) for n in small full; do [ -f "$c/build/$n/build_manifest.json" ] || return 1; done;;
  esac
}

donor_build() {               # newest earlier campaign whose build is valid for campaign $1 (prints its path)
  local c=$1 mode=${2:-strict} other   # mode loose (--no-build): same application and grids, intact executable
  for other in $(campaigns); do
    [ "$other" = "$c" ] && continue
    built "$other" || continue
    if "$python_bin" - "$c" "$other" "$mode" "$app" <<'PY'
import hashlib, json, sys
from pathlib import Path
new, old = (Path(p) for p in sys.argv[1:3])
loose = sys.argv[3] == 'loose'
a, b = (json.loads((p / 'campaign.json').read_text()) for p in (new, old))
keys = ('ops_commit', 'environment_sha256', 'builds', 'translator_python', 'application', 'senga_commit', 'hdf5',
        'senga_modified_files')
if loose:
    keys = ('builds', 'application')
# CloverLeaf reads the grid from clover.in at run time: one executable serves every grid, so the grids
# ('builds') do not decide whether an earlier build can be reused.
if sys.argv[4] == 'cloverleaf':
    keys = tuple(k for k in keys if k != 'builds')
same = all(a.get(k) == b.get(k) for k in keys)
clean = loose or (a.get('ops_uncommitted_changes') is False and b.get('ops_uncommitted_changes') is False)
code = (a.get('package') or {}).get('files_sha256', {}).get('sources.py')
same_code = loose or (code and code == (b.get('package') or {}).get('files_sha256', {}).get('sources.py'))
intact = True
for manifest in sorted((old / 'build').glob('*/build_manifest.json')):
    m = json.loads(manifest.read_text())
    exe = old / m['executable']
    intact = intact and exe.is_file() and hashlib.sha256(exe.read_bytes()).hexdigest() == m['sha256']
sys.exit(0 if same and clean and same_code and intact else 1)
PY
    then printf '%s\n' "$other"; return 0; fi
  done
  return 1
}

ensure_build() {
  local c=$1 donor
  local build_args=(build --campaign "$c")
  [ "$app" = senga2 ] && build_args+=(--build all)
  if built "$c"; then echo "Build: already complete in this campaign."; return; fi
  if [ "$nobuild" = 1 ]; then
    if [ "$dry" = 1 ]; then echo "Build: --no-build (dry run: no executable copied)."; return; fi
    donor=$(donor_build "$c" strict) || donor=$(donor_build "$c" loose) \
      || die "--no-build: no earlier $app campaign has an intact build$([ "$app" = cloverleaf ] || echo ' with the same grids'). Run without --no-build."
    rm -rf "$c/build"
    cp -a "$donor/build" "$c/build"
    "$python_bin" - "$c" "$donor" <<'PY'
import datetime, json, sys
from pathlib import Path
new, old = (Path(p) for p in sys.argv[1:3])
a, b = (json.loads((p / 'campaign.json').read_text()) for p in (new, old))
keys = ('ops_commit', 'ops_uncommitted_changes', 'environment_sha256', 'translator_python', 'senga_commit', 'hdf5',
        'senga_modified_files')
diff = {k: dict(this_campaign=a.get(k), executable_from=b.get(k)) for k in keys if a.get(k) != b.get(k)}
code = lambda c: (c.get('package') or {}).get('files_sha256', {}).get('sources.py')
if code(a) != code(b):
    diff['package_sources_py_sha256'] = dict(this_campaign=code(a), executable_from=code(b))
record = dict(reused_from=str(old), reason='--no-build', time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              identical=not diff, differences=diff)
(new / 'build_reused.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
print('Build: --no-build, reusing the executable of %s' % old.name)
if diff:
    print('WARNING: the executable was NOT built from the sources of this campaign. Differences:')
    for k, v in sorted(diff.items()):
        print('  %-26s this campaign=%s  executable=%s' % (k, v['this_campaign'], v['executable_from']))
    print('  (saved in %s)' % (new / 'build_reused.json'))
PY
  elif [ "$rebuild" = 0 ] && [ "$dry" = 0 ] && donor=$(donor_build "$c"); then
    echo "Build: reusing the executable of $(basename "$donor") (same OPS commit, environment and grids)."
    rm -rf "$c/build"
    cp -a "$donor/build" "$c/build"
  else
    [ "$rebuild" = 1 ] && echo "Build: compiling (--rebuild)." || echo "Build: compiling (no identical earlier build to reuse)."
  fi
  run bash "$c/launcher/run.sh" "${build_args[@]}"      # verifies a reused build; compiles otherwise
}

submitted() { [ -f "$1/submissions.json" ] && grep -q '"kind": *"sweep"' "$1/submissions.json"; }

submit() {
  local c=$1; shift
  local args=(submit --campaign "$c" --jobs "$jobs" --override-gonogo "$override")
  [ -n "$walltime" ] && args+=(--time "$walltime")
  run bash "$c/launcher/run.sh" "${args[@]}" "$@"
}

show_queue() { if command -v squeue >/dev/null 2>&1; then run squeue -u "$USER"; fi; }

# ------------------------------------------------------------------ list (all applications)
if [ "$action" = list ]; then
  echo "$app campaigns in $root (newest first):"
  campaigns | while read -r c; do echo "  $(basename "$c")"; done
  exit 0
fi

# ------------------------------------------------------------------ TGV top-3 (one script with options)
if [ "$app" = tgv-top3 ]; then
  tgv="$here/$package/run.sh"
  case "$action" in
    new)
      stamp=$(date -u +%Y%m%d_%H%M%S)
      run bash "$tgv" --submit --account "$account" --nodes "${nodes:-2,3,4,6,8,10,12,16,20,24,32,40}" \
          --date-stamp "$stamp" --time "${walltime:-01:00:00}"
      echo "Campaign: $root/$prefix$stamp";;
    continue|retry)
      c=$(pick_campaign)
      [ "$action" = retry ] && echo "NOTE: the TGV package has no retry of failed cases; continuing what is missing."
      read -r stamp nodes_c time_c < <("$python_bin" -c 'import json,sys
c = json.load(open(sys.argv[1] + "/campaign.json"))
print(c["stamp"], ",".join(map(str, c["nodes"])), c["time"])' "$c")
      run bash "$tgv" --submit --resume --account "$account" --nodes "$nodes_c" --date-stamp "$stamp" --time "$time_c";;
    status)
      c=$(pick_campaign)
      run bash "$tgv" --status --campaign "$c";;
  esac
  show_queue
  exit 0
fi

# ------------------------------------------------------------------ CloverLeaf / SENGA2 / TGV
case "$action" in
  new)
    prepare=(prepare --account "$account")
    [ -n "$grid" ] && prepare+=(--grid "$grid")
    [ -n "$nodes" ] && prepare+=(--nodes "$nodes")
    plan_extra=()
    if [ -n "$decomps" ] && [ "$app" != senga2 ]; then prepare+=(--decompositions "$decomps"); fi
    if [ -n "$decomps" ] && [ "$app" = senga2 ]; then      # SENGA2 selects by exclusion: leave out the rest
      scratch=$(mktemp -d)
      dry_plan=(plan --output-dir "$scratch")
      [ -n "$grid" ] && dry_plan+=(--grid "$grid")
      [ -n "$nodes" ] && dry_plan+=(--nodes "$nodes")
      bash "$here/$package/run.sh" "${dry_plan[@]}" >/dev/null
      excluded=$("$python_bin" - "$scratch/plan.json" "$decomps" <<'PY'
import json, sys
valid = [d['decomposition'] for d in json.load(open(sys.argv[1]))['decompositions']]
norm = lambda t: 'x'.join(str(int(v)) for v in t.lower().split('x'))
wanted = [norm(t) for t in sys.argv[2].split(',') if t.strip()]
bad = [w for w in wanted if w not in valid]
if bad or not wanted:
    sys.exit('ERROR: not in the SENGA2 decomposition set: %s\nValid: %s' % (', '.join(bad) or '(empty list)', ' '.join(valid)))
print(' '.join(d for d in valid if d not in wanted))
PY
      ) || { rm -rf "$scratch"; exit 2; }
      rm -rf "$scratch"
      for d in $excluded; do plan_extra+=(--exclude "$d"); done
    fi
    printf '+ bash %q' "$here/$package/run.sh"; printf ' %q' "${prepare[@]}"; printf '\n'
    if [ "$dry" = 1 ]; then echo "Dry run: the campaign is not created, so the next steps are not shown."; exit 0; fi
    out=$(bash "$here/$package/run.sh" "${prepare[@]}" | tee /dev/stderr)
    c=$(printf '%s\n' "$out" | sed -n 's/^Campaign: //p')
    [ -n "$c" ] || die "prepare did not report the campaign folder"
    ensure_build "$c"
    run bash "$c/launcher/run.sh" plan --campaign "$c" --steps "$steps" ${plan_extra[@]+"${plan_extra[@]}"}
    submit "$c"
    echo "Campaign: $c";;
  continue|retry)
    c=$(pick_campaign)
    echo "Campaign: $c"
    ensure_build "$c"
    [ -f "$c/plan.json" ] || run bash "$c/launcher/run.sh" plan --campaign "$c" --steps "$steps"
    extra=()
    if [ "$action" = retry ]; then extra+=(--resume --retry-failed); elif submitted "$c"; then extra+=(--resume); fi
    submit "$c" ${extra[@]+"${extra[@]}"};;
  status)
    c=$(pick_campaign)
    run bash "$c/launcher/run.sh" status --campaign "$c";;
esac
show_queue
