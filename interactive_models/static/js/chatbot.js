htmlToElement = function (html) {
    let template = document.createElement('template');
    html = html.trim(); // Never return a text node of whitespace as the result
    template.innerHTML = html;
    return template.content;
}

createMessage = function (message, bot) {
    let message_template = "";
    if (bot) {
        message_template += '<div class="message">';
        message_template += '  <img alt="" src="static/images/bear.jpg" />';
    } else {
        message_template += '<div class="message right">';
        message_template += '  <img alt="" src="static/images/user.png" />';
    }
    message_template += '  <div class="bubble">' + message;
    message_template += '    <div class="corner"></div>';
    message_template += '  </div>';
    message_template += '</div>';
    return message_template;
}


processUserInput = function (userInput) {
    let message = createMessage(userInput.value, false);
    console.log(message);
    const element = htmlToElement(message).firstChild;

    userInput.value = "";
    let chat_message = $('#chat_message')[0];
    chat_message.appendChild(element);
    chat_message.classList.add("animate");

    const margin_top = element.childNodes[3].offsetHeight - 25;
    element.childNodes[1].style = "margin-top:" + margin_top + "px";
    // chat_message.scrollIntoView({behavior: "smooth", block: "end", inline: "nearest"});
    chat_message.scrollTop = chat_message.scrollHeight;
}

processBotOutput = function (botMessage) {
    let message = createMessage(botMessage, true);
    const element = htmlToElement(message).firstChild;

    let chat_message = $('#chat_message')[0];
    chat_message.appendChild(element);

    margin_top = element.childNodes[3].offsetHeight - 25;
    element.childNodes[1].style = "margin-top:" + margin_top + "px";
    // chat_message.scrollIntoView({behavior: "smooth", block: "end", inline: "nearest"});
    chat_message.scrollTop = chat_message.scrollHeight;
}

// 设置登录界面
function setupLogin() {
    const loginForm = document.getElementById("loginForm"); // 登录表单
    const chatbox = document.getElementById("chatbox"); // 聊天框
    const anonymousLoginButton = document.getElementById("anonymousLogin"); // 游客模式按钮
    const signUpButton = document.getElementById("signUp"); // 注册按钮

    // 隐藏聊天框，显示登录界面
    chatbox.style.display = "none";
    document.getElementById("login").style.display = "block";

    // 清理并重新绑定游客模式按钮事件
    anonymousLoginButton.replaceWith(anonymousLoginButton.cloneNode(true)); // 替换按钮，清理所有事件监听器
    const newAnonymousLoginButton = document.getElementById("anonymousLogin");
    newAnonymousLoginButton.addEventListener("click", () => {
        console.log("游客模式 button clicked");
        document.getElementById("login").style.display = "none"; // 隐藏登录界面
        document.getElementById("chatbox").style.display = "block"; // 显示聊天框
        setupChat(); // 初始化聊天功能
    });

    // 清理并重新绑定注册按钮事件
    signUpButton.replaceWith(signUpButton.cloneNode(true)); // 替换按钮，清理所有事件监听器
    const newSignUpButton = document.getElementById("signUp");
    newSignUpButton.addEventListener("click", () => {
        console.log("注册 button clicked");
        // 处理注册逻辑
    });

    // 登录表单提交事件
    loginForm.addEventListener("submit", function (event) {
        event.preventDefault(); // 阻止表单提交

        // 获取用户名和密码
        const username = loginForm.querySelector('input[type="text"]').value;
        const password = loginForm.querySelector('input[type="password"]').value;

        // 模拟登录逻辑
        chatbox.style.display = "block"; // 显示聊天框
        document.getElementById("login").style.display = "none"; // 隐藏登录界面
        loginForm.reset();
    });
}

// 设置聊天功能
function setupChat() {
    const chatbox = document.getElementById("chatbox");

    // 检查并确保 .chat-messages 存在
    let messagesContainer = chatbox.querySelector(".chat_messages");
    if (!messagesContainer) {
        console.warn(".chat-messages element not found. Creating one...");
        messagesContainer = document.createElement("div");
        messagesContainer.className = "chat-messages";
        chatbox.appendChild(messagesContainer);
    }
    let userInput = chatbox.querySelector(".chat-input");
    let userInputButton = chatbox.querySelector(".chat-input-button");
    let closeButton = chatbox.querySelector("#close");

    // 清空聊天框状态
    messagesContainer.innerHTML = ""; // 清空聊天内容
    userInput.value = ""; // 清空输入框内容
    let timestamp = Date.now(); // Timestamp
    let dialogues = []; // 对话列表

    // 解绑旧的事件监听器（通过替换节点）
    userInputButton.replaceWith(userInputButton.cloneNode(true));
    const newUserInputButton = chatbox.querySelector(".chat-input-button");

    userInput.replaceWith(userInput.cloneNode(true));
    const newUserInput = chatbox.querySelector(".chat-input");

    closeButton.replaceWith(closeButton.cloneNode(true));
    const newCloseButton = chatbox.querySelector("#close");

    // 绑定发送按钮事件
    newUserInputButton.addEventListener("click", () => {
        handleButtonClick(newUserInput.value);
        handleUserInput(timestamp, newUserInput, dialogues, function (updatedDialogues) {
            dialogues = updatedDialogues; // Update dialogues in setupChat
            console.log("Dialogues in setupChat after update:", dialogues);
        });
    });

    // 绑定输入框的 Enter 键事件
    newUserInput.addEventListener("keyup", (event) => {
        if (event.keyCode === 13) {
            // Enter 键
            event.preventDefault();
            handleUserInput(timestamp, newUserInput, dialogues, function (updatedDialogues) {
                dialogues = updatedDialogues; // Update dialogues in setupChat
                console.log("Dialogues in setupChat after update:", dialogues);
            });
        }
    });

    // 绑定关闭按钮事件
    newCloseButton.addEventListener("click", () => {
        console.log("Close button clicked");
        resetChat();
        setupLogin(); // 返回登录界面
    });

    // 重置聊天框
    function resetChat() {
        // 清空聊天内容
        messagesContainer.innerHTML = "";
        newUserInput.value = "";
        chatbox.style.display = "none";
        document.getElementById("login").style.display = "block"; // 显示登录界面
    }
}

function handleUserInput(timestamp, userInput, dialogues, callback) {
    let parameters = {
        "timestamp": timestamp,
        "message": userInput.value,
        "dialogues": dialogues
    };
    console.log(userInput.value);
    processUserInput(userInput);

    let postRequest = {
        type: 'post',
        url: '/event/chat',
        contentType: 'application/json',
        data: JSON.stringify(parameters),
        dataType: 'json',
        success: function (result) {
            dialogues.splice(0, dialogues.length, ...result.dialogues); // Modify dialogues in place
            processBotOutput(result.message);
            if (callback) {
                callback(dialogues);
            }
        },
        error: function (result) {
            console.log(result);
        }
    };
    $.ajax(postRequest);
}

// Function to handle button clicks
function handleButtonClick(buttonName) {
    console.log(`${buttonName} button clicked`);
    // You can add additional logic here, such as form submission or navigation
}


