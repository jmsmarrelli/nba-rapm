"""Render readable article tables from full-precision result CSVs."""
from pathlib import Path
import csv, math, re
ROOT=Path(__file__).resolve().parents[1]
LABELS={
 'att':'Attempts','team_seasons':'Team-seasons','r':'Correlation','r2':'R²',
 'r2_gain':'Change in R²','r2_with_interaction':'R² with interaction',
 'offence_coef':'Offense coefficient','defence_coef':'Defense coefficient',
 'interaction_coef':'Interaction coefficient','games_in_pbp':'Games in source',
 'keep_rate':'Retained share','mean_minute_error':'Mean minute error (minutes)',
 'excluded_special_ft_points':'Special FT points excluded','recorded_points_kept':'Recorded points',
 'mae':'MAE','rmse':'RMSE','mse':'MSE','log_loss':'Log loss','brier':'Brier score',
 'auc':'ROC AUC','spec':'Specification','r2_vs_average':'R² vs league mean',
 'gain_over_team':'MSE improvement vs team baseline','training_selected_alpha':'Training-selected penalty',
 'd_total':'Total change','d_location':'Location component','d_conversion':'Conversion component',
 'scaled_effect':'Estimate','scaled_lo':'95% lower','scaled_hi':'95% upper','p':'p-value',
 'n':'Observations','n_games':'Games','lo':'95% lower','hi':'95% upper',
 'beta':'Coefficient','se':'Standard error','t':'t statistic','zone5':'Shot zone',
 'slope_per_10pp':'Slope per 10 percentage points','se_per_10pp':'Standard error',
}
VALUES={'geo':'Geometry model','full':'Recorded-context model','constant':'Constant-probability baseline'}
COUNTS={'team_seasons','games','games_in_pbp','games_kept','recorded_points_kept',
 'excluded_special_ft_points','modeled_points','test_games','n','n_games','pairs',
 'att','shots','seasons','close_games','assists','made_fg','nights'}
def cell(value, column):
    value=VALUES.get(value,value)
    try: number=float(value)
    except (ValueError,TypeError): return str(value).replace('|',r'\|').replace('\n',' ')
    if not math.isfinite(number): return '—'
    if column=='season' and 2000<number<2100 and number.is_integer():
        year=int(number);return f'{year-1}–{year%100:02}'
    if column in COUNTS and number.is_integer():return f'{int(number):,}'
    if column in {'p','r2_gain'} and 0<abs(number)<.001:return f'{number:.2g}'
    precision=4 if column in {'r2','r2_gain','r2_with_interaction','r2_vs_average','p'} else 3
    if abs(number)<.5*10**(-precision):number=0.
    return f'{number:.{precision}f}'
def table(match):
    fields=match.group(1).split('|');path=ROOT/'output'/fields[0]
    with path.open(newline='') as f:
        reader=csv.DictReader(f);rows=list(reader);cols=fields[1].split(',') if len(fields)>1 else reader.fieldnames
    if len(fields)>2:
        filters=fields[2].split(',')
        for condition in filters:
            col,value=condition.split('=',1);rows=[r for r in rows if r[col]==value]
    missing=set(cols)-set(reader.fieldnames)
    if missing:raise ValueError(f'{path}: unknown columns {missing}')
    lines=['| '+' | '.join(LABELS.get(c,c.replace('_',' ').capitalize()) for c in cols)+' |',
           '|'+'|'.join(['---']*len(cols))+'|']
    for row in rows:lines.append('| '+' | '.join(cell(row[c],c) for c in cols)+' |')
    lines.append('\nSource: ['+fields[0]+'](output/'+fields[0]+').')
    return '\n'.join(lines)
def main():
    text=(ROOT/'docs/article_template.md').read_text()
    text=re.sub(r'{{table:([^}]+)}}',table,text)
    if '{{table:' in text:raise ValueError('Unresolved table template')
    (ROOT/'README.md').write_text(text)
if __name__=='__main__':main()
