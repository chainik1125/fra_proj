"""Ensure splitting changes scheduling, not the search or its tie-breaking."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'baseline_lib'))
import rp8
from single_eval import choose_single_settings,signed_grid
from steering import DEFAULT_PROTOCOL


class ShardMergeTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.c=Path(self.tmp.name);self.jobs=[]
        candidates=[{'method':'single','features':[i],'selection_score':float(50-i)} for i in range(50)]
        self.grid=[{'candidate':c,'alpha':a,'metrics':{'triggered_to_clean_js_bits':
            .1 if c['features'][0] in (1,4) and a==2 else .9}}
            for c in candidates for a in signed_grid(DEFAULT_PROTOCOL['alphas'])]
        for i in range(4):
            run=self.c/f'shard{i}';run.mkdir();self.jobs.append({'id':f'8x-shard{i}','state':'complete','run':str(run)})
            for name,value in [('selection',{'candidates':candidates}),('split_manifest',{'keys':[1,2]}),('validation_baseline',{'metrics':{'jsd':.99}})]:
                (run/f'{name}.json').write_text(json.dumps(value))
            (run/'validation.jsonl').write_text('\n'.join(json.dumps(r) for r in self.grid if r['candidate']['features'][0]%4==i)+'\n')

    def tearDown(self):self.tmp.cleanup()

    def test_merge_preserves_full_search_and_original_positive_tie_order(self):
        self.assertEqual(rp8.merge_shards(self.c,8,self.jobs),choose_single_settings(self.grid))

    def test_duplicate_record_is_rejected(self):
        p=Path(self.jobs[0]['run'])/'validation.jsonl';p.write_text(p.read_text()+p.read_text().splitlines()[0]+'\n')
        with self.assertRaises(AssertionError):rp8.merge_shards(self.c,8,self.jobs)

    def test_different_baseline_is_rejected(self):
        p=Path(self.jobs[1]['run'])/'validation_baseline.json';p.write_text('{"metrics":{"jsd":0.98}}')
        with self.assertRaises(AssertionError):rp8.merge_shards(self.c,8,self.jobs)


if __name__=='__main__':unittest.main()
