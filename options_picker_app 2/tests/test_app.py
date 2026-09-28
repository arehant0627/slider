"""Run the Streamlit app headlessly in demo mode and click through it."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit.testing.v1 import AppTest

at = AppTest.from_file(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"), default_timeout=180)
at.run()
assert not at.exception, at.exception
assert at.sidebar.toggle[0].value is True            # no key -> demo data
assert any("Find trades" in i.value for i in at.info)
print("CHECK first load: demo mode on, instructions shown, no errors")

at.sidebar.number_input[0].set_value(100_000)
at.sidebar.number_input[1].set_value(0.5)
at.sidebar.button[0].click()
at.run()
assert not at.exception, at.exception
assert any("Target reached" in s.value for s in at.success), [s.value for s in at.success] + [w.value for w in at.warning]
labels = [m.label for m in at.metric]
assert "Weekly target" in labels and "Average loss, worst 5% of weeks" in labels
print("CHECK run at 0.5%: target reached;", {m.label: m.value for m in at.metric})
assert len(at.dataframe) >= 1
print("  trades table rows:", len(at.dataframe[0].value))

# $ target that can't be reached: warning, still a plan
at.sidebar.radio[0].set_value("$")
at.run()
at.sidebar.number_input[1].set_value(50_000)
at.sidebar.button[0].click()
at.run()
assert not at.exception, at.exception
assert any("out of reach" in w.value for w in at.warning), [w.value for w in at.warning]
print("CHECK impossible $ target: warns with the maximum and still shows a plan")
print("ALL APP CHECKS PASSED")
