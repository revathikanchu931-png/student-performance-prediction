document.addEventListener('DOMContentLoaded',()=>{
  const form=document.querySelector('#prediction-form');if(!form)return;
  const errorBox=document.querySelector('#form-error');const result=document.querySelector('#result');const submit=form.querySelector('button[type="submit"]');
  form.addEventListener('submit',async(event)=>{
    event.preventDefault();errorBox.classList.add('hidden');errorBox.textContent='';result.classList.add('hidden');
    if(!form.reportValidity())return;
    const data=Object.fromEntries(new FormData(form).entries());
    for(const key of ['study_hours','attendance','previous_score','assignment_completion','sleep_hours'])data[key]=Number(data[key]);
    submit.disabled=true;submit.textContent='Estimating…';
    try{
      const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
      const body=await response.json();if(!response.ok){throw new Error((body.details||[body.error||'Please check your entries.']).join(' '));}
      result.innerHTML='';
      const card=document.createElement('div');card.className='result-card';
      const score=document.createElement('div');score.className='result-score';score.innerHTML=`<div><strong>${body.predicted_score}</strong><small>estimated score out of 100</small></div>`;
      const copy=document.createElement('div');copy.className='result-copy';
      const pill=document.createElement('span');pill.className='category-pill';pill.textContent=body.category;
      const title=document.createElement('h2');title.textContent=body.student_name?`A snapshot for ${body.student_name}`:'Your learning snapshot';
      const explanation=document.createElement('p');explanation.textContent=body.explanation;
      const tipsTitle=document.createElement('h3');tipsTitle.textContent='Ideas to consider';const list=document.createElement('ul');list.className='tips';
      body.suggestions.forEach(tip=>{const li=document.createElement('li');li.textContent=tip;list.appendChild(li);});
      const caveat=document.createElement('p');caveat.className='result-disclaimer';caveat.textContent='This is an estimate from synthetic training data, not a guaranteed examination result.';
      copy.append(pill,title,explanation,tipsTitle,list,caveat);card.append(score,copy);result.appendChild(card);result.classList.remove('hidden');result.scrollIntoView({behavior:'smooth',block:'center'});
    }catch(error){errorBox.textContent=error.message;errorBox.classList.remove('hidden');}
    finally{submit.disabled=false;submit.innerHTML='Predict my score <span>→</span>';}
  });
});
