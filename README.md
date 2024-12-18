# Create Module


# Register Component
確定 Module 有放入 `modules` 資料夾的分類中  
若有要傳遞參數，可用 `__init__` 傳入參數  
```py
from src.modules.collect.camera_module import CameraModule
...
app: App = App(
    modules=[
        CameraModule(some_args),

        # 重複的Module
        CameraModule(some_args), # Same Class Module ❌❌❌
        CameraModule(some_args).set_key('camera_1'), # Same Class Module ✔️✔️✔️
    ],
)
```
- 如果有同Class的Module，可以使用 .set_key() 來設定Key，沒有設定Key的話，會是 "default"

# Module Lifecycle hooks
按照觸發順序列出
1. __init__(self) -> None
Class 初始化的時候 ``App(modules=[ ... ]) 時就會觸發``  
此時請不要進行 GetComponent 的操作，因為還沒有初始化完畢
- 傳參數、初始化變數

2. __mount__(self) -> None
App 初始化完畢後，會觸發所有 Module 的 __mount__ (按照 modules 順序)
此時可以進行 GetComponent 的操作
- 初始化變數、 綁定其他 Module、 載入模型

3. __sysready__(self) -> None
! 只有 App.run() 後才會觸發 (僅執行一次)
當所有 Module 都初始化Mount完畢後，會觸發所有 Module 的 __sysready__
此時可以進行 GetComponent 的操作
- 只有打開程式後才會觸發的咚咚


4. __unmount__(self) -> None
- - 模組被卸載時，可以清理資源

# 週期性執行
只會在__mount__後才會開始執行
TickRate 不同，App.max_tick 可調整
- update(self):
- fixed_update(self):


# 調用其他 Module 互相溝通
❌❌❌ 不要在 __init__ 中調用其他 Module，因為可能還沒有初始化完畢
```py
def __init__(self):
    self.camera_module = self.app.get_component(CameraModule) # 可能得到 None

```
✔️✔️✔️ 在 __mount__ 中調用其他 Module
```py
def __mount__(self):
    self.camera1_module = self.app.get_component(CameraModule)
    self.camera2_module = self.app.get_component(CameraModule, 'camera_2') # 取得指定Key的CameraModule
    self.depth_module = self.app.get_component(DepthModule)
def update(self):
    self.camera.do_something()
    frame = self.camera_module.frame
    depth = self.depth_module.depth
```


# Data Viewer
使用 decorator 來顯示資料
- interval : 每幾秒更新一次

```py
class CameraModule(BaseModule):
    ...
    @on_view_update(interval=1 / 30)
    def custom_name(self, providers: Providers) -> None:
        if self.frame is not None:
            # https://rerun.io/docs/reference/types/archetypes/image
            providers.rerun.log("Camera", rr.Image(self.frame))
        else:
            self.logger.warning("No frame to display.")
```