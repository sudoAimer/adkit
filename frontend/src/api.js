// 同源 API 客户端，将网络异常和接口校验错误转成中文反馈。
export async function request(path, options = {}) {
  // JSON 请求统一编码；读取错误体时保留服务端的具体失败原因。
  let response;
  try {
    response = await fetch(`/api${path}`, {
      ...options,
      headers: { "Content-Type": "application/json", ...options.headers },
    });
  } catch {
    throw new Error("无法连接服务，请确认后端已启动后重试。");
  }
  if (response.status === 204) return null;
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail =
      typeof data.detail === "string"
        ? data.detail
        : "提交参数有误，请检查输入。";
    throw new Error(detail);
  }
  return data;
}

export function uploadImages(taskId, kind, files, onProgress) {
  // 使用浏览器上传事件显示真实传输进度，不将传输百分比混为模型进度。
  return new Promise((resolve, reject) => {
    const body = new FormData();
    for (const file of files) body.append("files", file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/tasks/${taskId}/images/${kind}`);
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable)
        onProgress(Math.round((event.loaded / event.total) * 100));
    };
    xhr.onload = () => {
      let data;
      try {
        data = JSON.parse(xhr.responseText);
      } catch {
        reject(new Error("服务返回了无效响应"));
        return;
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else
        reject(
          new Error(
            typeof data.detail === "string" ? data.detail : "上传失败，请重试",
          ),
        );
    };
    xhr.onerror = () => reject(new Error("上传连接中断，请重试"));
    xhr.send(body);
  });
}
