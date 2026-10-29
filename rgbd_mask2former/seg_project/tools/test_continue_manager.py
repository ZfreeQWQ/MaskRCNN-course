import io
import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
import run_continue_training as manager


class ManagerTests(unittest.TestCase):
    def test_vendor_memory_parser_requires_known_devices(self):
        text='| 0 Iluvatar BI-V150 |\n| 68MiB / 32768MiB |\n| 1 Iluvatar BI-V150 |\n| 2000MiB / 32768MiB |'
        self.assertEqual(manager.parse_gpu_memory(text),{0:68,1:2000})
        self.assertEqual(manager.parse_gpu_memory('unrecognized output'),{})

    def scenario(self,export_finished):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); prior=Path.cwd()
            smoke=root/'seg_project/experiments/mask2former/d_continue6_smoke_20261026/job.exit_code'
            smoke.parent.mkdir(parents=True); smoke.write_text('0')
            output=root/'training'; result=root/'round'; result.mkdir()
            if export_finished: (result/'test_complete.json').write_text('{}')
            first=Mock(); second=Mock()
            first.poll.side_effect=[None,None] if export_finished else [None,0]
            first.wait.return_value=143 if export_finished else 0
            first.pid=101; second.wait.return_value=0; second.pid=202
            states=[dict(attempt=1),dict(attempt=2)]
            launches=[(first,io.StringIO(),states[0]),(second,io.StringIO(),states[1])]
            try:
                os.chdir(root)
                with patch.object(manager,'ROOT',output),patch.object(manager,'ROUND',result),\
                     patch.object(manager,'launch',side_effect=launches) as launch,\
                     patch.object(manager,'spare_gpus_idle',return_value=True) as idle,\
                     patch.object(manager,'last_checkpoint',side_effect=[None,'model_0000901.pth','model_0000901.pth']),\
                     patch.object(manager.time,'sleep'):
                    manager.main()
                    if export_finished:
                        self.assertEqual(launch.call_args_list[1].args,([0,1,2,3],True,2))
                        first.send_signal.assert_called_once_with(signal.SIGTERM)
                        self.assertTrue((output/'gpu_expansion.json').exists())
                    else:
                        launch.assert_called_once_with([2,3],False,1)
                        idle.assert_not_called()
                        first.send_signal.assert_not_called()
            finally: os.chdir(prior)

    def test_expansion_waits_for_export(self): self.scenario(False)

    def test_expansion_resumes_on_four_gpus_at_new_checkpoint(self): self.scenario(True)


if __name__=='__main__': unittest.main()
